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
    def __init__(self, feature_dim, dilate_kernel=3, threshold=0.5):
        super(TemporalSampling, self).__init__()
        self.threshold = threshold
        self.dilate_kernel = dilate_kernel
        self.padding = (dilate_kernel - 1) // 2

        self.mlp = nn.Linear(feature_dim, 1)
        # self.relu = nn.ReLU()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, feats, record_len):
        confidence_score = []

        batch_feats = self.regroup(feats, record_len)
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


"""
class CorrectionNet(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024, # -1,
        'remove_borders': 4,
    }
    def __init__(self, feature_dim, dilate_kernel=3, threshold=0.5):
        super(CorrectionNet, self).__init__()
        self.config = {**self.default_config}

        self.threshold = threshold
        self.dalite_kernel = dilate_kernel
        self.padding = (dalite_kernel - 1) // 2

        self.mlp = nn.Linear(feature_dim, 1)
        # self.relu = nn.ReLU()

        self.convPa = nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(feature_dim // 2, 1, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(feature_dim // 2, self.config['descriptor_dim'], kernel_size=1, stride=1, padding=0)

    def forward(self, feats):
        L, C, H, W = feats.shape
        data = self.extract(feats)
        keypoints, _, descriptors = **self.to_tensor(data)

        kpts0, kpts1 = keypoints[:,0,:,:].view(1, -1, H, W), keypoints[:,1:,:,:].view(L-1, -1, H, W)
        desc0, desc1 = descriptors[:,0,:,:], descriptors[:,1:,:,:]

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        
        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def to_tensor(self, data):
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        return data

    def extract(self, x):
        # Compute the dense keypoint scores
        cPa = F.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]

        b, _, h, w = scores.shape
        scores = scores.squeeze(1)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [torch.nonzero(s > self.config['keypoint_threshold']) for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[self.remove_borders(k, s, self.config['remove_borders'], h*8, w*8) for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[self.top_k_keypoints(k, s, self.config['max_keypoints']) for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = F.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = F.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0] for k, d in zip(keypoints, descriptors)]

        return keypoints, scores, descriptors

    def remove_borders(self, keypoints, scores, border: int, height: int, width: int):
        "" Removes keypoints too close to the border ""
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
        mask = mask_h & mask_w
        return keypoints[mask], scores[mask]

    def top_k_keypoints(self, keypoints, scores, k: int):
        if k >= len(keypoints):
            return keypoints, scores
        scores, indices = torch.topk(scores, k, dim=0)
        return keypoints[indices], scores
"""

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
            # self.corr_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttenFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
                # corr_network = CorrectionNet(num_filters[idx])
                # self.corr_modules.append(corr_network)
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
                    confidence_maps = self.sampling(x, record_len)
                    x = x * confidence_maps
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

