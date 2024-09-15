"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler

Intermediate fusion for camera based collaboration
"""
import numpy as np
from einops import rearrange, repeat

import torch
from torch import nn
import torch.nn.functional as F

from opencood.models.vd_modules.sensor_blocks import ImgCamEncode
from opencood.models.vd_modules.camera_encode_blocks import CamEncode
from opencood.models.vd_modules.utils import basic, vox, geom


class CoVQMF3(nn.Module):  
    def __init__(self, args):
        super(CoVQMF3, self).__init__()
        img_args = args['img_unproj']
        self.grid_conf = img_args['grid_conf']   # 网格配置参数
        self.data_aug_conf = img_args['data_aug_conf']   # 数据增强配置参数
        self.downsample = img_args['img_downsample']  # 下采样倍数
        self.bevC = img_args['bev_dim']  # 图像特征维度
        
        # 用于投影到BEV视角的参数
        voxels_size = torch.LongTensor([int((row[1] - row[0]) / row[2] + 0.5) for row in [self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound']]])
        self.X = voxels_size[0]  # 256
        self.Y = voxels_size[1]  # 256
        self.Z = voxels_size[2]  # 16

        scene_centroid = torch.from_numpy(np.array([0.0, 0.0, 0.0]).reshape([1, 3])).float()

        bounds = (self.grid_conf['xbound'][0], self.grid_conf['xbound'][1],
                  self.grid_conf['ybound'][0], self.grid_conf['ybound'][1],
                  self.grid_conf['zbound'][0], self.grid_conf['zbound'][1])
        # bounds = (-52, 52, -52, 52, 0, 6)

        self.vox_util = vox.Vox_util(self.Z, self.Y, self.X, scene_centroid=scene_centroid, bounds=bounds, assert_cube=False)
        self.camencode = CamEncode(self.bevC, self.downsample)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, image_inputs_dict, mode=[0,1,2]):   # loss: 5.91->0.76
        x, intrins, extrins = image_inputs_dict['imgs'], image_inputs_dict['intrins'], image_inputs_dict['extrins']
        B, N, C, imH, imW = x.shape     # torch.Size([4, N, 3, 320, 480])

        __p = lambda x: basic.pack_seqdim(x, B)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, B)

        img_feature = self.camencode(x)
        img_feature = rearrange(img_feature, 'b n c h w -> (b n) c h w')
        img_feature = self.feature2vox_simple(img_feature, __p(intrins), __p(extrins), __p, __u)
        img_feature = rearrange(img_feature, 'b c h w z-> b (c z) h w')
        
        return img_feature
        

    def feature2vox_simple(self, features, pix_T_cams, cam0_T_camXs, __p, __u):
        pix_T_cams_ = pix_T_cams
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs)
        B, C, Hf, Wf = features.shape
        
        sy = Hf / float(self.data_aug_conf['final_dim'][0])
        sx = Wf / float(self.data_aug_conf['final_dim'][1])

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # pix_T_cams_ shape: [6,4,4]  feature down sample -> featpix_T_cams_

        feat_mems_ = self.vox_util.unproject_image_to_mem(features, 
            basic.matmul2(featpix_T_cams_, camXs_T_cam0_), camXs_T_cam0_, 
            self.Z, self.Y, self.X)

        # feat_mems_ shape： torch.Size([6, 128, 200, 8, 200])
        feat_mems = __u(feat_mems_)  # B, S, C, Z, Y, X # torch.Size([1, 6, 128, 200, 8, 200])

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)  # B, C, Z, Y, X
        feat_mem = feat_mem.permute(0, 1, 4, 3, 2) # [0, ...].unsqueeze(0) # ZYX -> XYZ
        return feat_mem
