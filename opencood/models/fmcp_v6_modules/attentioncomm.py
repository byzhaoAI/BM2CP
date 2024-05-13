# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modified from Yue Hu
# Author: Anonymous

import math
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

"""
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
        'max_keypoints': 256, # -1,
        'match_threshold': 0.2,
    }
    def __init__(self, feature_dim, ratio=1):
        super(Correction, self).__init__()
        self.config = {**self.default_config}
        self.ratio = ratio

        self.convS = nn.Conv2d(feature_dim, 1, kernel_size=3, stride=1, padding=1)
        # self.convSa = nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1)
        # self.convSb = nn.Conv2d(feature_dim // 2, 1, kernel_size=1, stride=1, padding=0)

        self.convD = nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1)
        # self.convDa = nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1)
        # self.convDb = nn.Conv2d(feature_dim // 2, 1, kernel_size=1, stride=1, padding=0)

        self.gnn = AttentionalGNN(feature_dim=self.config['max_keypoints'])

        self.proj = nn.Conv1d(feature_dim // 2, feature_dim // 2, kernel_size=3, stride=1, padding=1)
        self.projS = nn.Conv1d(feature_dim // 2, feature_dim // 2, kernel_size=1)

        self.final_proj = nn.Conv1d(feature_dim // 2, 3, kernel_size=1, bias=True)

    def forward(self, feats):

        L, _, H, W = feats.shape
        #start_time = time.time()
        
        # Compute the scores [L,1,H,W] and descriptors[L,C/2,H,W]
        scores = F.sigmoid(self.convS(feats))
        descriptors = F.relu(self.convD(feats))

        scores = scores.squeeze(1) # -> b,h,w
        scores = self.simple_nms(scores, self.config['nms_radius'])
        scores = scores.view(L, -1)

        # Keep the k keypoints with highest score
        _scores = []
        if self.config['max_keypoints'] >= 0:
            scores, indices = torch.topk(scores, self.config['max_keypoints'], dim=1)

        # Compute the dense descriptors
        descriptors = F.normalize(descriptors, p=2, dim=1)
        descriptors = descriptors.flatten(2) # -> b,c/2,h*w
        _descriptors = []
        for i in range(L):
            _descriptors.append(descriptors[i, :, indices[i]])
        descriptors = torch.stack(_descriptors)

        # Multi-layer Transformer network.
        descriptors = self.gnn(descriptors)
        descriptors = F.relu(self.proj(descriptors))
        scores = self.projS(descriptors)

        _, indices = scores.sort(dim=-1, descending=True)
        descriptors = descriptors.gather(dim=-1, index=indices)

        # Final MLP projection.
        descriptors = self.final_proj(descriptors)

        descriptors = descriptors - descriptors[0:1,:,:]
        # ----------------------------------------------------------
        # min_desc, indices = torch.min(descriptors.abs(), dim=2)
        # min_desc = min_desc * self.ratio

        signal = descriptors / descriptors.abs()
        indices = torch.argmin(descriptors.abs(), dim=2, keepdim=True)
        #print(indices.shape, indices)
        min_desc = (descriptors.gather(dim=2, index=indices) * signal.gather(dim=2, index=indices)).squeeze(2)
        # print(min_desc.shape, min_desc)
        if torch.isnan(min_desc).any():
            zeros_mask = torch.zeros_like(min_desc).to(min_desc.device)
            min_desc = torch.where(torch.isnan(min_desc), zeros_mask, min_desc)
        # min_desc shape: L,3
        # ----------------------------------------------------------

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


class ImportanceSampling(nn.Module):
    def __init__(self, feature_dim, dilate_kernel=1, threshold=0.5):
        super(ImportanceSampling, self).__init__()
        # self.att = ScaledDotProductAttention(feature_dim)
        self.threshold = threshold
        self.dilate_kernel = dilate_kernel
        self.padding = (dilate_kernel - 1) // 2

        self.mlp = nn.Linear(feature_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, node_feature):
        # node_features: (L, CV H, W)
        L, C, H, W = node_feature.shape
        # (H*W, cav_num, C), perform attention on each pixel.
        if L > 1:
            node_feature = rearrange(node_feature, 'l c h w -> (h w) l c')
            ego_node_feature = node_feature[:, 0, :].view(-1, 1, C)
        
            neighbor_node_feature = node_feature[:, 1:L, :] #.view(-1, L-1, C)
            score = self.att_forward(ego_node_feature.repeat(1,L-1,1), neighbor_node_feature, neighbor_node_feature, C)
            score = self.relu(self.mlp(score)).sigmoid()
            
            if self.dilate_kernel > 1:
                ones_mask = torch.ones_like(score).to(score.device)
                zeros_mask = torch.zeros_like(score).to(score.device)
                mask = torch.where(score > self.threshold, ones_mask, zeros_mask)
                mask = F.max_pool2d(mask, kernel_size=self.dilate_kernel, stride=1, padding=self.padding)
            else:
                mask = torch.where(score > self.threshold, 1, 0)
            
            ego_mask = torch.ones((H*W, 1, 1)).to(ego_node_feature.device)
            overall_mask = torch.concat([ego_mask, mask], dim=1)        
            node_feature = node_feature * overall_mask

            node_feature = rearrange(node_feature, '(h w) l c-> l c h w', h=H, w=W)
        
        return node_feature

    def att_forward(self, query, key, value, C):
        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(C)
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


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


class ImportanceFusion(nn.Module):
    def __init__(self, feature_dim, ratio=1, dilate_kernel=1, threshold=0.5):
        super(ImportanceFusion, self).__init__()
        # self.att = ScaledDotProductAttention(feature_dim)
        self.ratio = True if ratio > 0 else False
        if ratio > 0:
            self.correction = Correction(feature_dim, ratio=ratio)
        self.sampling = ImportanceSampling(feature_dim, dilate_kernel=dilate_kernel, threshold=threshold)
        self.fusion = ShapleyFusion(feature_dim)

    def forward(self, node_feature):
        # correct relative pose error
        if self.ratio:
            node_feature = self.correction(node_feature)

        # sampling
        node_feature = self.sampling(node_feature)
        
        # fusion
        node_feature = self.fusion(node_feature)
        return node_feature


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
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = ImportanceFusion(num_filters[idx], ratio=args['corr_ratio'], dilate_kernel=args['dilate_kernel'], threshold=args['threshold'])
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

                #if i==0:
                #    batch_confidence_maps = self.regroup(rm, record_len)
                #    _, communication_masks, communication_rates = communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                #    x = x * communication_masks

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

