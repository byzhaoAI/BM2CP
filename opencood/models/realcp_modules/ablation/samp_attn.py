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
            mask = torch.where(score > 0.5, ones_mask, zeros_mask)
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

            self.sampling = TemporalSampling(num_filters[0])
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttenFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
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

