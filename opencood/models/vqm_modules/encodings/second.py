""" 
Author: Co-VQM
"""
import numpy as np
import torch
import torch.nn as nn

from opencood.models.common_modules.mean_vfe import MeanVFE
from opencood.models.common_modules.sparse_backbone_3d import VoxelBackBone8x


class SECOND(nn.Module):   
    def __init__(self, args, device):
        super(SECOND, self).__init__()
        # for second
        # self.batch_size = args['batch_size']
        cav_lidar_range = args['lidar_range']
        voxel_size = args['voxel_size']
        grid_size = (np.array(cav_lidar_range[3:6]) - np.array(cav_lidar_range[0:3])) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)

        # mean_vfe
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'], 4, grid_size)

        out_channels = args['backbone_3d']['num_features_out'] * 2
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, data_dict, training=False):
        batch_dict = {
            'voxel_features': data_dict['processed_lidar2']['voxel_features'],
            'voxel_coords': data_dict['processed_lidar2']['voxel_coords'],
            'voxel_num_points': data_dict['processed_lidar2']['voxel_num_points'],
            'batch_size': torch.sum(data_dict['record_len']).cpu().numpy(),
            'record_len': data_dict['record_len']
        }
        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        # height compression
        batch_dict = self.height_compression(batch_dict)

        spatial_features = self.tconv(batch_dict['spatial_features'])
        return spatial_features

    def height_compression(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
