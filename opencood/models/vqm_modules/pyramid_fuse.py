# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.vqm_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.vqm_modules.resblock import ResNetModified, Bottleneck, BasicBlock


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x



def warp_affine_simple(src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False):

    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)



def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)
    
    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)
        
    record_len : list
        shape: (B)
        
    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3) 
    """

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego), 
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device), 
                                    scores_in_ego)

        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    
    return out


class PyramidFusion(ResNetBEVBackbone):
    # def __init__(self, model_cfg, max_modality_agent_index, agent_types, fp, align_last, input_channels=64):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        # self.max_modality_agent_index = max_modality_agent_index
        # self.fp = fp
        # self.align_last = align_last

        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        
        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

    def forward_single(self, spatial_features, ith_agent, align_block=None):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        
        proj_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            level_features = feature_list[i]
            if align_block is not None:
                level_features = eval(f"align_block.a{ith_agent+1}_backward_proj{i}").weight.unsqueeze(-1).unsqueeze(-1) * level_features
            proj_feature_list.append(level_features)
            
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)

        # final_feature = self.decode_multiscale_feature(feature_list)
        final_feature = self.decode_multiscale_feature(proj_feature_list)

        return final_feature, occ_map_list
    
    def forward_collab(self, spatial_features, record_len, affine_matrix, align_block, fp, align_last, max_modality_agent_index):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            ## ** original code ** ##
            if fp and align_last:
                level_features = self.align(i, feature_list[i], record_len, align_block, max_modality_agent_index)
            else:
                level_features = feature_list[i]
            ## ** no embedding code ** ##
            # level_features = feature_list[i]

            occ_map = eval(f"self.single_head_{i}")(level_features)  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            fused_feature_list.append(weighted_fuse(level_features, score, record_len, affine_matrix, self.align_corners))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)
        
        return fused_feature, occ_map_list

    def align(self, level, batch_features, record_len, align_block, max_modality_agent_index):
        split_features = regroup(batch_features, record_len)
        
        out = []
        # iterate each batch
        for agent_num, split_feature in zip(record_len, split_features):
            features = []
            for ith_agent in range(agent_num):
                if ith_agent == max_modality_agent_index:
                    features.append(split_feature[ith_agent:ith_agent+1])
                else:
                    features.append(
                        eval(f"align_block.a{ith_agent+1}_backward_proj{level}").weight.unsqueeze(-1).unsqueeze(-1) * split_feature[ith_agent:ith_agent+1]
                    )

            out.append(
                torch.cat(features, dim=0)
            )

        out = torch.cat(out, dim=0)
        return out
