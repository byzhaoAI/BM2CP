# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opencood.models.heal_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.heal_modules.resblock import ResNetModified, Bottleneck, BasicBlock


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
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
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

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list
    
    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list = None, cam_crop_info = None):
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
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list] 
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}

        sum_len, m_len, x_C, x_H, x_W = spatial_features.shape
        spatial_features = rearrange(spatial_features, 'b m c h w -> (b m) c h w')
        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map = rearrange(occ_map, '(b m) c h w -> b m c h w', b=sum_len, m=m_len)
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask

            feature = rearrange(feature_list[i], '(b m) c h w -> b m c h w', b=sum_len, m=m_len)
            fused_feature = self.weighted_fuse(feature, score, record_len, affine_matrix, self.align_corners)
            # (B,m_len,C,H,W) -> (B*m_len,C,H,W)
            fused_feature = rearrange(fused_feature, 'b m c h w -> (b m) c h w')
            fused_feature_list.append(fused_feature)
            # fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))

        fused_feature = self.decode_multiscale_feature(fused_feature_list)
        # fused_feature: (b*m_len,c,h,w); occ_map_list: {(sum_len,m_len, 1, H, W)}
        return fused_feature, occ_map_list 

    def weighted_fuse(self, x, score, record_len, affine_matrix, align_corners):
        """
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), m_len, C, H, W)
        
        score : torch.Tensor
            score, (sum(n_cav), m_len, 1, H, W)
            
        record_len : list
            shape: (B)
            
        affine_matrix : torch.Tensor
            normalized affine matrix from 'normalize_pairwise_tfm'
            shape: (B, L, L, 2, 3) 
        """

        _, m_len, C, H, W = x.shape
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
            feature_in_ego = self.warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W), align_corners=align_corners)
            scores_in_ego = self.warp_affine_simple(split_score[b], t_matrix[i, :, :, :], (H, W), align_corners=align_corners)

            scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
            scores_in_ego = torch.softmax(scores_in_ego, dim=0)
            scores_in_ego = torch.where(torch.isnan(scores_in_ego), 
                                        torch.zeros_like(scores_in_ego, device=scores_in_ego.device), 
                                        scores_in_ego)

            # feature_in_ego: (L,m_len,C,H,W); scores_in_ego: (L,m_len,1,H,W) -> (m_len,C,H,W)
            out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))

        # {(m_len,C,H,W)} -> (B,m_len,C,H,W)
        out = torch.stack(out)
        
        return out

    def warp_affine_simple(self, src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False):

        # M: (L,2,3)
        # src: (L,m_len,C,H,W)

        L, m_len, C, H, W = src.size()

        src = rearrange(src, 'l m c h w -> (l m) c h w')
        M = rearrange(M.unsqueeze(1).repeat(1,m_len,1,1), 'l m x y -> (l m) x y')
        grid = F.affine_grid(M, [L*m_len, C, dsize[0], dsize[1]], align_corners=align_corners).to(src)
        warpped_src = F.grid_sample(src, grid, align_corners=align_corners)
        warpped_src = rearrange(warpped_src, '(l m) c h w -> l m c h w', l=L, m=m_len)
        return warpped_src
