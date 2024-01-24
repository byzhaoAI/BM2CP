import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from opencood.models.how2comm_modules.feature_flow import FlowGenerator, ResNetBEVBackbone
# from opencood.models.comm_modules.mutual_communication import Communication


class How2commPreprocess(nn.Module):
    def __init__(self, args, channel, delay):
        super(How2commPreprocess, self).__init__()
        self.flow_flag = args['flow_flag'] 
        self.channel = channel
        self.frame = args['fusion_args']['frame']  
        self.delay = delay  
        self.flow = FlowGenerator(args)

        self.commu_module = Communication(
            args['fusion_args']['communication'], in_planes=self.channel)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def get_grid(self, flow):
        m, n = flow.shape[-2:]
        shifts_x = torch.arange(
            0, n, 1, dtype=torch.float32, device=flow.device)
        shifts_y = torch.arange(
            0, m, 1, dtype=torch.float32, device=flow.device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

        grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
        workspace = torch.tensor(
            [(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

        flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

        return flow_grid

    def resample(self, feats, flow):
        flow_grid = self.get_grid(flow)
        warped_feats = F.grid_sample(
            feats, flow_grid, mode="bilinear", padding_mode="border")

        return warped_feats

    def communication(self, feats, record_len, history_list, confidence_map_list):
        feat_list = self.regroup(feats, record_len)
        sparse_feat_list, commu_loss, commu_rate, sparse_mask = self.commu_module(
            feat_list,confidence_map_list)
        sparse_feats = torch.cat(sparse_feat_list, dim=0)
        sparse_history_list = []
        for i in range(len(sparse_feat_list)):
            sparse_history = torch.cat([history_list[i][:1], sparse_feat_list[i][1:]], dim=0)
            sparse_history_list.append(sparse_history)
        sparse_history = torch.cat(sparse_history_list, dim=0)
        return sparse_feats, commu_loss, commu_rate, sparse_history

    def forward(self, feat_curr, feat_history, record_len, backbone=None, heads=None):
        feat_curr = self.regroup(feat_curr, record_len)
        B = len(feat_curr)
        feat_list = [[] for _ in range(B)]
        for bs in range(B):
            feat_list[bs] += [feat_curr[bs], feat_history[bs]]

        if self.flow_flag:
            feat_final, offset_loss = self.flow(feat_list)
        else:
            offset_loss = torch.zeros(1).to(record_len.device)
            x_list = []
            for bs in range(B):
                delayed_colla_feat = feat_list[bs][self.delay][1:]  
                ego_feat = feat_list[bs][0][:1]  
                x_list.append(
                    torch.cat([ego_feat, delayed_colla_feat], dim=0))
            feat_final = torch.cat(x_list, dim=0)

        return feat_final, offset_loss


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.thre = args['thre']
        self.compressed_dim = args['compressed_dim']
        if 'gaussian_smooth' in args:
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, psm):
        B = len(psm)
        _, _, H, W = psm[0].shape
        

        private_confidence_maps = []
        private_communication_masks = []
        communication_rates = []
        
        for b in range(B):
            ori_private_communication_maps = psm[b].sigmoid().max(dim=1)[0].unsqueeze(1)  
            
            if self.smooth:
                private_communication_maps = self.gaussian_filter(ori_private_communication_maps)
            else:
                private_communication_maps = ori_private_communication_maps
            private_confidence_maps.append(private_communication_maps)  
                

            ones_mask = torch.ones_like(private_communication_maps).to(private_communication_maps.device)
            zeros_mask = torch.zeros_like(private_communication_maps).to(private_communication_maps)
            
            private_mask = torch.where(private_communication_maps > self.thre, ones_mask, zeros_mask)  
            cav_num = private_mask.shape[0]
            private_rate = private_mask[1:].sum()/((cav_num-1) * H * W)
            
            private_mask_nodiag = private_mask.clone()
            ones_mask = torch.ones_like(private_mask).to(private_mask.device)
            private_mask_nodiag[::2] = ones_mask[::2]  
            private_communication_masks.append(private_mask_nodiag)
            communication_rates.append(private_rate)
            
        communication_rates = sum(communication_rates)/B
        private_mask = torch.cat(private_communication_masks, dim=0)  
        
        return private_mask, communication_rates, private_confidence_maps