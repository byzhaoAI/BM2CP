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

from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

"""
class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, query, x):
        cav_num, C, H, W = x.shape
        query = query.view(1, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        # torch.Size([12600, 2, 64]) torch.Size([25200, 2, 64])
        x = self.att(query, x, x)
        x = x.permute(1, 2, 0).view(1, C, H, W)[0]  # C, W, H before
        return x
"""

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


class TemporalSampling(nn.Module):
    def __init__(self, feature_dim, frame, decay_rate=0.5, threshold=0.5):
        super(TemporalSampling, self).__init__()
        self.frame = frame + 1
        history_weight = []
        for i in range(frame + 1):
            history_weight.append((decay_rate)**i)
        self.history_weight = F.softmax(torch.Tensor(history_weight), 0)
        self.threshold = threshold
        
        self.mlp = nn.Linear(feature_dim, 1)
        # self.relu = nn.ReLU()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, history_data_list, record_len):
        assert len(history_data_list) == self.frame, f"The length of list {len(history_data_list)} is not consistent with {self.frame}"
        
        confidence_score = []
        for i, history_data in enumerate(history_data_list):
            batch_history_data = self.regroup(history_data, record_len)
            
            for j, hist_feat in enumerate(batch_history_data):
                # (H*W, cav_num, C), perform attention on each pixel.
                L, C, H, W = hist_feat.shape
                node_feature = rearrange(hist_feat, 'l c h w -> (h w) l c')
                ego_node_feature = node_feature[:, 0, :].view(-1, 1, C)
            
                if L > 1:
                    neighbor_node_feature = node_feature[:, 1:, :].view(-1, L-1, C)
                    score = self.att_forward(ego_node_feature.repeat(1,L-1,1), neighbor_node_feature, neighbor_node_feature, C)
                    score = self.mlp(score).sigmoid()
                    ego_score = torch.ones((H*W, 1, 1)).to(ego_node_feature.device)
                    overall_score = torch.concat([ego_score, score], dim=1)        
                    #node_feature = node_feature * overall_mask
                else:
                    overall_score = torch.ones((H*W, 1, 1)).to(ego_node_feature.device)
                overall_score = rearrange(overall_score, '(h w) l c-> l c h w', h=H, w=W)

                if i == 0:
                    confidence_score.append(overall_score * self.history_weight[i].to(overall_score.device))
                else:
                    confidence_score[j] = confidence_score[j] + overall_score * self.history_weight[i].to(overall_score.device)
        
        confidence_mask = []
        
        for score in confidence_score:
            ones_mask = torch.ones_like(score).to(score.device)
            zeros_mask = torch.zeros_like(score).to(score.device)
            mask = torch.where(score > 0.5, ones_mask, zeros_mask)
            confidence_mask.append(mask)
        confidence_mask = torch.concat(confidence_mask, dim=0)
        # dilate
        confidence_mask = F.max_pool2d(confidence_mask, kernel_size=3, stride=1, padding=1)
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

            self.sampling = TemporalSampling(num_filters[0], args['frame'])
            self.fuse_modules = nn.ModuleList()
            self.corr_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttenFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
                self.corr_modules.append(nn.Conv2d(num_filters[idx], 2, kernel_size=3, stride=1, padding=1, bias=True))
        
        else:
            self.sampling = TemporalSampling(args['agg_operator']['feature_dim'], args['frame'])
            self.fuse_modules = AttenFusion(args['agg_operator']['feature_dim'])
            self.corr_modules = nn.Conv2d(args['agg_operator']['feature_dim'], 2, kernel_size=3, stride=1, padding=1, bias=True)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x_list, semantic_maps, record_len, pairwise_t_matrix, backbone=None, heads=None):
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
        x = x_list[0]

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
                hist_feats_list = []
                for hist in x_list:
                    hist_feats_list.append(backbone.resnet(hist))
                
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                scale_hist_feats_list = [hist_feats[i] if with_resnet else backbone.blocks[i](hist_feats) for hist_feats in hist_feats_list]

                if i==0:
                    confidence_maps = self.sampling(scale_hist_feats_list, record_len)
                    x = x * confidence_maps

                    # batch_confidence_maps = self.regroup(confidence_maps, record_len)
                    # _, communication_masks, communication_rates = communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                    # x = x * communication_masks
                
                # flow_grid = rearrange(self.corr_modules[i](x), 'b l h w -> b h w l')
                # corr_x = F.grid_sample(x, flow_grid, mode="bilinear", padding_mode="border")
                # batch_node_features = self.regroup(corr_x, record_len)
                
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
                    
                    #_neighbor_feature = neighbor_feature * batch_correction_maps[b]
                    #neighbor_feature = torch.stack([neighbor_feature[0], _neighbor_feature[1]])
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
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
