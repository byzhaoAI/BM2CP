import torch
import torch.nn as nn
import numpy as np
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor

from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization

from opencood.models.hmvit_modules.spatial_transformation import SpatialTransformation
from opencood.models.hmvit_modules.base_camera_lidar_intermediate import BaseCameraLiDARIntermediate
from opencood.models.hmvit_modules.naive_decoder import NaiveDecoder
from opencood.models.hmvit_modules.hetero_decoder import HeteroDecoder
from opencood.models.hmvit_modules.hetero_fusion import HeteroFusionBlock
from opencood.models.hmvit_modules.base_transformer import HeteroFeedForward

from einops import rearrange

import torchvision

from opencood.models.vqm_modules.f1 import CoVQMF1
from opencood.models.vqm_modules.encodings.second import SECOND as BaseSECOND


class Multimodal(nn.Module):
    def __init__(self, args):
        super(Multimodal, self).__init__()
        
        self.basebone = CoVQMF1(args['multimodal'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

    def forward(self, batch_dict, mode=[0,1], training=False):
        # rec_loss, svd_loss, bfp_loss = torch.tensor(0.0, requires_grad=True).to(record_len.device), torch.tensor(0.0, requires_grad=True).to(record_len.device), torch.tensor(0.0, requires_grad=True).to(record_len.device)
        
        f, _, _, _rec_loss, _svd_loss = self.basebone(batch_dict, mode=mode, training=training)
        features = self.backbone({'spatial_features': f})['spatial_features_2d']

        if self.shrink_flag:
            features = self.shrink_conv(features)
        return features, _rec_loss, _svd_loss


class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()
        pillar_args = args['pillar']
        self.pillar_vfe = PillarVFE(pillar_args['pillar_vfe'],
                                num_point_features=4,
                                voxel_size=pillar_args['voxel_size'],
                                point_cloud_range=pillar_args['lidar_range'])
        cav_lidar_range = pillar_args['lidar_range']
        voxel_size = pillar_args['voxel_size']
        grid_size = (np.array(cav_lidar_range[3:6]) - np.array(cav_lidar_range[0:3])) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        pillar_args['point_pillar_scatter']['grid_size'] = grid_size
        self.scatter = PointPillarScatter(pillar_args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

    def forward(self, batch_dict):
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        features = self.backbone(batch_dict)['spatial_features_2d']

        if self.shrink_flag:
            features = self.shrink_conv(features)
        return features


class SECOND(nn.Module):
    def __init__(self, args):
        super(SECOND, self).__init__()
        self.second = BaseSECOND(args['second'], device=args['device'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False
        

    def forward(self, data_dict):
        f = self.second(data_dict)
        features = self.backbone({'spatial_features': f})['spatial_features_2d']

        if self.shrink_flag:
            features = self.shrink_conv(features)
        return features
