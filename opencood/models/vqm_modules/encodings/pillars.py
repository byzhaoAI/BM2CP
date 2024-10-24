"""
Author: Anonymous
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter


class Pillars(nn.Module):   
    def __init__(self, args, device):
        super(Pillars, self).__init__()
        # lidar 分支网络
        #（1）PillarVFE              pcdet/models/backbones_3d/vfe/pillar_vfe.py   # 3D卷积, 点特征编码
        #（2）PointPillarScatter     pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py   # 2D卷积，创建（实际就是变形）一个大小为(C，H，W)的伪图像
        cav_lidar_range = args['lidar_range']
        voxel_size = args['voxel_size']
        grid_size = (np.array(cav_lidar_range[3:6]) - np.array(cav_lidar_range[0:3])) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        args['point_pillar_scatter']['grid_size'] = grid_size

        self.pillar_vfe = PillarVFE(args['pillar_vfe'], num_point_features=4, voxel_size=args['voxel_size'], point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

    def forward(self, data_dict, training=False):
        batch_dict = {
            'voxel_features': data_dict['processed_lidar']['voxel_features'],
            'voxel_coords': data_dict['processed_lidar']['voxel_coords'],
            'voxel_num_points': data_dict['processed_lidar']['voxel_num_points'],
            'batch_size': torch.sum(data_dict['record_len']).cpu().numpy(),
            'record_len': data_dict['record_len']
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        lidar_feature = batch_dict['spatial_features']

        return lidar_feature
