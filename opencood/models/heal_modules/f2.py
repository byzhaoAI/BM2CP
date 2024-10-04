""" 
Author: Co-VQM
"""

import torch
import torch.nn as nn

from opencood.models.common_modules.mean_vfe import MeanVFE
from opencood.models.common_modules.sparse_backbone_3d import VoxelBackBone8x


class CoVQMF2(nn.Module):   
    def __init__(self, args):
        super(CoVQMF2, self).__init__()
        # mean_vfe
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'], 4, args['grid_size'])

        out_channels = args['backbone_3d']['num_features_out'] * 2
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch_dict):
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
