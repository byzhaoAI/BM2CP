"""
A model zoo for intermediate fusion.
Please make sure your pairwise_t_matrix is normalized before using it.
Enjoy it.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from icecream import ic


def warp_affine_simple(src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False):

    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


class DomainEncoder(nn.Module):
    def __init__(self, inplanes, planes):
        super(DomainEncoder, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
  
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        identity = self.bn2(self.conv2(x))
        out += identity
        out = self.relu(out)
        out = self.conv3(out)
        return out

class SpatialEncoder2(nn.Module):
    def __init__(self, inplanes, planes):
        super(SpatialEncoder2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
  
        self.conv2 = nn.Conv2d(inplanes, 1, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out2 = self.bn2(self.conv2(x))
        #out3 = torch.max(x, dim=1).values
        out3 = torch.max(x, dim=1, keepdim=True).values
        out = out + out2 + out3

        return out


class DimReduction(nn.Module):
    def __init__(self, inplanes, planes, norm_layer=None):
        super(DimReduction, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class DomainGeneralizedFeatureFusion3(nn.Module):
    def __init__(self, single_dims, vis_feats):
        super(DomainGeneralizedFeatureFusion3, self).__init__()
        self.fused_dims = single_dims * 2
        print('using DGFF3 fusion without RT matrix.')

        # Doamin generalized feature  encoder
        self.domain_encoder = DomainEncoder(self.fused_dims, self.fused_dims//8)
        self.spatial_encoder = SpatialEncoder2(self.fused_dims, self.fused_dims//8)
        # Dimension reduction 
        self.dim_reduction = DimReduction(self.fused_dims, single_dims)

        self.down_vehicle = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.down_inf = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(self.fused_dims, 2, kernel_size=3, padding=1, bias=False)
        self.gate = SimpleGate(single_dims, 2)

        self.vis_feats = vis_feats

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch

        for b in range(B):
            x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                # algin feature
                x1 = self.down_vehicle(x1)
                x2 = self.down_inf(x2)
                flow = self.flow_make(torch.cat([x1, x2], 1))
                flow = self.gate(x1, flow)
                x2 = self.flow_warp(x2, flow, size=(H, W))
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                # domain attention
                domain_attention = self.domain_encoder(concat) #(1, 2*C, H, W)
                domain_attention = F.softmax(domain_attention.view(1, C, 2, H, W), dim = 2)
                
                spatial_attention = self.spatial_encoder(concat).view(1, 1, 1, H, W)  #(1, 1, H, W)

                fused_feature = torch.mul(domain_attention*spatial_attention, concat.view(1, C, 2, H, W)).view(1,-1, H, W)
                # dimension reduction 
                x = self.dim_reduction(fused_feature)
            out.append(x[0])

        out = torch.stack(out)

        if self.vis_feats:
            return  out, batch_node_features[0]
        else:
            return out
    
    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class SumFusion_multiscale(nn.Module):
    def __init__(self, args):
        super(SumFusion_multiscale, self).__init__()
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)

        print('using SumFusion_multiscale, with RT matrix.')

    def forward(self, x, record_len, pairwise_t_matrix, backbone=None):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        out = []
        feat_list = []
        
        if self.multi_scale:
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                batch_node_features = regroup(x, record_len)
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4) # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
                    x_fuse.append(torch.sum(neighbor_feature, dim=0))
                x_fuse = torch.stack(x_fuse)
                feat_list.append(x_fuse)
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
            return  x_fuse, feat_list

        else:
            split_x = regroup(x, record_len)
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                i = 0 # ego
                #(N, C, H, W)
                neighbor_feature = warp_affine_simple(split_x[b],
                                                t_matrix[i, :, :, :],
                                                (H, W))
                # (N, C, H, W)
                feature_fused = torch.sum(neighbor_feature, dim=0)
                out.append(feature_fused)

            return torch.stack(out)


class SumFusion_multiscale2(nn.Module):
    def __init__(self, args):
        super(SumFusion_multiscale2, self).__init__()
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'MAX':
                    fuse_network = MaxFusion2()
                self.fuse_modules.append(fuse_network)
        print('using SumFusion_multiscale2, without RT matrix.')

    def forward(self, x, record_len, pairwise_t_matrix, backbone=None):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        out = []
        feat_list = []
        
        if self.multi_scale:
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                batch_node_features = regroup(x, record_len)
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    x_fuse.append(self.fuse_modules[i](node_features))
                x_fuse = torch.stack(x_fuse)
                feat_list.append(x_fuse)
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
            return  x_fuse, feat_list

        else:
            split_x = regroup(x, record_len)
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                i = 0 # ego
                # (N, C, H, W)
                feature_fused = torch.sum(split_x[b], dim=0)
                out.append(feature_fused)

            return torch.stack(out)
