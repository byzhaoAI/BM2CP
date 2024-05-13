# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modified from Yue Hu
# Author: Anonymous


from turtle import update
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from copy import deepcopy
import time
import math

from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.realcp_modules.mamba import VisionMamba


def communication(batch_confidence_maps, record_len, pairwise_t_matrix):
    # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
    # pairwise_t_matrix: (B,L,L,2,3)
    # thre: threshold of objectiveness
    # a_ji = (1 - q_i)*q_ji
    B, L, _, _, _ = pairwise_t_matrix.shape
    _, _, H, W = batch_confidence_maps[0].shape

    communication_masks = []
    communication_rates = []
    for b in range(B):
        N = record_len[b]

        ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(
            1)  # dim1=2 represents the confidence of two anchors

        communication_maps = ori_communication_maps

        ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
        zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
        communication_mask = torch.where(communication_maps > 0.5, ones_mask, zeros_mask)

        communication_rate = communication_mask[0].sum() / (H * W)

        communication_mask_nodiag = communication_mask.clone()
        ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
        communication_mask_nodiag[::2] = ones_mask[::2]

        communication_masks.append(communication_mask_nodiag)
        communication_rates.append(communication_rate)

    communication_rates = sum(communication_rates) / B
    communication_masks = torch.concat(communication_masks, dim=0)
    # print(communication_masks.shape, torch.count_nonzero(communication_masks))
    return {}, communication_masks, communication_rates


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        qkv = rearrange(x, 'b c l -> l b c')
        message = self.attn(qkv, qkv.transpose(1,2), qkv, C)
        message = rearrange(message, 'l b c -> b c l')
        return x + message

    def attn(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, C): # -> Tuple[torch.Tensor,torch.Tensor]:
        scores = torch.einsum('bnh,bhm->bnm', query, key) / C**.5
        prob = F.softmax(scores, dim=-1)
        return torch.einsum('bnm,bmh->bnh', prob, value)


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        # layers_name: : List[str] 
        super().__init__()
        self.layer = AttentionalPropagation(feature_dim)

    def forward(self, desc):
        delta = self.layer(desc)
        return desc + delta


class Correction(nn.Module):
    default_config = {
        'nms_radius': 4,
        'max_keypoints': 1024, # -1,
        'match_threshold': 0.2,
    }
    def __init__(self, feature_dim, corr_ratio=1):
        super(Correction, self).__init__()
        self.config = {**self.default_config}
        self.corr_ratio = corr_ratio

        self.convPa = nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(feature_dim // 2, 1, kernel_size=1, stride=1, padding=0)

        self.gnn = AttentionalGNN(feature_dim=self.config['max_keypoints'])

        self.final_proj = nn.Conv1d(feature_dim // 2, 3, kernel_size=1, bias=True)

        self.bin_score = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, feats):
        #start_time = time.time()
        
        # Compute the dense keypoint scores
        descriptors = F.relu(self.convPa(feats))
        scores = self.convPb(descriptors)
        scores = F.sigmoid(scores)
        # scores = F.softmax(scores, 1)

        scores = scores.squeeze(1) # -> b,h,w
        scores = self.simple_nms(scores, self.config['nms_radius'])
        scores = scores.view(scores.shape[0], -1)

        # Keep the k keypoints with highest score
        _scores = []
        if self.config['max_keypoints'] >= 0:
            scores, indices = torch.topk(scores, self.config['max_keypoints'], dim=1)

        # Compute the dense descriptors
        descriptors = F.normalize(descriptors, p=2, dim=1)
        descriptors = descriptors.view(descriptors.shape[0], descriptors.shape[1], -1)
        _descriptors = []
        for i in range(descriptors.shape[0]):
            _descriptors.append(descriptors[i, :, indices[i]])
        descriptors = torch.stack(_descriptors)

        # Multi-layer Transformer network.
        descriptors = self.gnn(descriptors)

        # Final MLP projection.
        descriptors = self.final_proj(descriptors)
        descriptors = descriptors - descriptors[0:1,:,:]
        min_desc, _ = torch.min(descriptors, dim=2)
        min_desc = min_desc * self.corr_ratio

        t_matrix = []
        for _desc in min_desc:
            t_matrix.append(
                torch.Tensor([
                    [math.cos(_desc[2]), -math.sin(_desc[2]), _desc[0]], 
                    [math.sin(_desc[2]), math.cos(_desc[2]) , _desc[1]]
                ]).to(feats.device)
            )
        t_matrix = torch.stack(t_matrix)

        grid = F.affine_grid(t_matrix, feats.size())
        feats = F.grid_sample(feats, grid)
        #print('correct time: ', time.time() - start_time)
        return feats

    def simple_nms(self, scores, nms_radius: int):
        # Fast Non-maximum suppression to remove nearby points
        assert(nms_radius >= 0)

        def max_pool(x):
            return F.max_pool2d(x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)


class ShapleyFusion(nn.Module):
    def __init__(self, feature_dim):
        super(ShapleyFusion, self).__init__()

        self.contrib_func = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.ReLU(),
        )
        # self.mlp = nn.Linear(feature_dim, 1)
        # self.relu = nn.ReLU()
        
        # self.k = nn.Linear(feature_dim, 1)
        # self.b = nn.Linear(feature_dim, 1)

    def forward(self, feats):
        L, C, H, W = feats.shape
        
        """
        feats = rearrange(feats, 'l c h w -> (h w) l c')
        local_Q = self.mlp(feats)
        agent_matrix = torch.eye(L).unsqueeze(0).unsqueeze(-1).to(feats.device) # (1, L, L, 1) 
        total_Q_i = []
        for i in range(L):
            total_Q_i.append(torch.sum(local_Q * (1 - agent_matrix[:, i, :, :]), dim=1, keepdim=False))
        total_Q_i = torch.stack(total_Q_i, dim=1) - local_Q

        ones_mask = torch.ones_like(total_Q_i).to(total_Q_i.device)
        zeros_mask = torch.zeros_like(total_Q_i).to(total_Q_i.device)
        # mask = torch.where(total_Q_i > 0.5, ones_mask, zeros_mask)
        mask = torch.where(total_Q_i > 0, ones_mask, zeros_mask)

        weight_cross_agent = (total_Q_i * mask).softmax(dim=1)
        feats = torch.sum(feats * weight_cross_agent, dim=1, keepdim=False)
        feats = rearrange(feats, '(h w) c-> c h w', h=H, w=W)
        """

        feats = rearrange(feats, 'l c h w -> l (h w) c')
        
        shapley_values = torch.zeros((L,H*W,1)).to(feats.device)
        for idx in range(L):
            # all subsets not contains i-th agent
            set_without_i = list(range(L))
            set_without_i.remove(idx)
            subset_without_i = self.generate_subsets(set_without_i)

            contrib_feats_i = self.contrib_func(feats[idx])
            for subset in subset_without_i:
                cardinal = len(subset)
                
                contrib_feats_with_i = feats[subset + [idx]].view(cardinal+1,H*W,C)
                contrib_feats_with_i = self.contrib_func(torch.sum(contrib_feats_with_i, dim=0, keepdim=False))
                contrib = contrib_feats_with_i - contrib_feats_i

                weight = (math.factorial(cardinal) * math.factorial(L-cardinal-1) / math.factorial(L)) # Weight = |S|!(n-|S|-1)!/n!
                shapley_values[idx] = shapley_values[idx] + weight * contrib
            
            # Add the term corresponding to the empty set
            shapley_values[idx] = shapley_values[idx] + contrib_feats_i / L
            
        feats = torch.sum(feats * shapley_values.softmax(dim=0), dim=0, keepdim=False)
        feats = rearrange(feats, '(h w) c-> c h w', h=H, w=W)
        return feats

    def generate_subsets(self, lst):  
        # 计算子集的总数（2的列表长度次方减去1，因为不包括空集）  
        num_subsets = 2 ** len(lst) - 1
        # 使用位运算生成所有子集  
        subsets = [[lst[i] for i, subset_bit in enumerate(bin(subset)[2:].zfill(len(lst))) if subset_bit == '1'] for subset in range(1, num_subsets + 1)]
        return subsets


class TemporalSampling(nn.Module):
    def __init__(self, feature_dim, dilate_kernel=3, threshold=0.5):
        super(TemporalSampling, self).__init__()
        self.threshold = threshold
        self.dilate_kernel = dilate_kernel
        self.padding = (dilate_kernel - 1) // 2

        self.attn = VisionMamba(
            #image_size=(96,352),
            patch_size=16, 
            stride=8, 
            embed_dim=feature_dim, 
            depth=1, 
            expand=1,
            rms_norm=True, 
            residual_in_fp32=True, 
            fused_add_norm=True, 
            final_pool_type='mean', 
            if_abs_pos_embed=True, 
            if_rope=False, 
            if_rope_residual=False, 
            bimamba_type="v2", 
            if_cls_token=True, 
            if_devide_out=True, 
            use_middle_cls_token=True
        )
        print(self.attn)

        self.mlp = nn.Linear(feature_dim, 1)
        # self.relu = nn.ReLU()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, feats, record_len):
        confidence_score = []

        batch_feats = self.regroup(feats, record_len)
        """
        for j, feats in enumerate(batch_feats):
            # (H*W, cav_num, C), perform attention on each pixel.
            L, C, H, W = feats.shape
            node_feature = rearrange(feats, 'l c h w -> (h w) l c')
            ego_node_feature = node_feature[:, 0, :].view(-1, 1, C)

            if L > 1:
                neighbor_node_feature = node_feature[:, 1:, :].view(-1, L-1, C)
                score = self.att_forward(ego_node_feature.repeat(1,L-1,1), neighbor_node_feature, neighbor_node_feature, C)
                score = self.mlp(score).sigmoid()
                ego_score = torch.ones((H*W, 1, 1)).to(ego_node_feature.device)
                batch_score = torch.concat([ego_score, score], dim=1)        
                #node_feature = node_feature * overall_mask
            else:
                batch_score = torch.ones((H*W, 1, 1)).to(ego_node_feature.device)
            batch_score = rearrange(batch_score, '(h w) l c-> l c h w', h=H, w=W)
            confidence_score.append(batch_score)
        """
        for j, feats in enumerate(batch_feats):
            L, C, H, W = feats.shape
            if L > 1:
                score = self.attn(rearrange(feats, 'l c h w -> l (h w) c'), return_features=True)  # BCHW -> BNC
                score = self.mlp(score).sigmoid()
                ego_score = torch.ones((1,H*W,1)).to(feats.device)
                batch_score = torch.concat([ego_score, score[1:,:,:]], dim=0)
            else:
                batch_score = torch.ones((1,H*W,1)).to(feats.device)
            batch_score = rearrange(batch_score, 'l (h w) c-> l c h w', h=H, w=W)
            confidence_score.append(batch_score)

        confidence_mask = []
        for score in confidence_score:
            ones_mask = torch.ones_like(score).to(score.device)
            zeros_mask = torch.zeros_like(score).to(score.device)
            mask = torch.where(score > self.threshold, ones_mask, zeros_mask)
            confidence_mask.append(mask)
        confidence_mask = torch.concat(confidence_mask, dim=0)
        # dilate
        confidence_mask = F.max_pool2d(confidence_mask, kernel_size=self.dilate_kernel, stride=1, padding=self.padding)
        return confidence_mask

    def att_forward(self, query, key, value, C):
        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(C)
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

        
class AttenComm(nn.Module):
    def __init__(self, args):
        super(AttenComm, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)

            self.sampling = TemporalSampling(num_filters[0], dilate_kernel=args['dilate_kernel'], threshold=args['threshold'])
            self.fuse_modules = nn.ModuleList()
            self.corr_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = ShapleyFusion(num_filters[idx])
                # fuse_network = AttenFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
                corr_network = Correction(num_filters[idx], corr_ratio=args['corr_ratio'])
                # corr_network = CorrectionModule(num_filters[idx])
                self.corr_modules.append(corr_network)
        else:
            self.fuse_modules = AttenFusion(args['agg_operator']['feature_dim'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, rm, record_len, pairwise_t_matrix, backbone=None, heads=None):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        if self.multi_scale:
            ups = []
            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)

                #if x.shape[-1] != rm.shape[-1]:
                #    rm = F.interpolate(rm, size=x.shape[2:], mode='bilinear', align_corners=False)

                if i==0:
                    #start_time = time.time()
                    confidence_maps = self.sampling(x, record_len)
                    x = x * confidence_maps
                    #print('sampling time: ', time.time() - start_time)
                    
                    # batch_confidence_maps = self.regroup(rm, record_len)
                    # _, communication_masks, communication_rates = communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                    # x = x * communication_masks

                # split x:[(L1, C, H, W), (L2, C, H, W), ...]
                # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
                batch_node_features = self.regroup(x, record_len)
                
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4)
                    # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))

                    neighbor_feature = self.corr_modules[i](neighbor_feature)
                    
                    #start_time = time.time()
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                    #print('fuse time: ', time.time() - start_time)
                x_fuse = torch.stack(x_fuse)

                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
                
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
            
            communication_rates = 0
        else:
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len)
            batch_confidence_maps = self.regroup(rm, record_len)

            _, communication_masks, communication_rates = communication(batch_confidence_maps, record_len, pairwise_t_matrix)

            x_fuse = []
            for b in range(B):
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                t_matrix[0, :, :, :],
                                                (H, W))
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
        
        return x_fuse, communication_rates, {}

