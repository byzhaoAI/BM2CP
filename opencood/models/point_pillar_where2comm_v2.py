# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
# from numpy import record
import torch
import torch.nn as nn

from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.dcn_net import DCNNet

from opencood.models.where2comm_v2_modules.utils import basic, vox, geom
from opencood.models.where2comm_v2_modules.sensor_blocks import ImgCamEncode
from opencood.models.where2comm_v2_modules.where2comm_attn import Where2comm
from opencood.models.where2comm_v2_modules.base_bev_backbone_resnet import ResNetBEVBackbone

from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple


class PointPillarWhere2commV2(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commV2, self).__init__()
        self.proj_first = args['proj_first']
        self.max_cav = args['max_cav']
        
        # camera 分支网络
        img_args = args['img_params']
        self.grid_conf = img_args['grid_conf']   # 网格配置参数
        self.data_aug_conf = img_args['data_aug_conf']   # 数据增强配置参数
        self.downsample = img_args['img_downsample']  # 下采样倍数
        
        voxels_size = torch.LongTensor([int((row[1] - row[0]) / row[2] + 0.5) for row in [self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound']]])
        self.X = voxels_size[0]  # 256
        self.Y = voxels_size[1]  # 256
        self.Z = voxels_size[2]  # 16

        # scene_centroid_x, scene_centroid_y, scene_centroid_z
        scene_centroid = torch.from_numpy(np.array([0.0, 0.0, 0.0]).reshape([1, 3])).float()

        bounds = (self.grid_conf['xbound'][0], self.grid_conf['xbound'][1],
                  self.grid_conf['ybound'][0], self.grid_conf['ybound'][1],
                  self.grid_conf['zbound'][0], self.grid_conf['zbound'][1])
        # bounds = (-52, 52, -52, 52, 0, 6)

        self.vox_util = vox.Vox_util(self.Z, self.Y, self.X, scene_centroid=scene_centroid, bounds=bounds, assert_cube=False)
        # self.vox_util = vox.Vox_util(self.Z, self.Y, self.X, scene_centroid=self.scene_centroid, bounds=self.bounds, position = self.opt.position, length_pose_encoding = length_pose_encoding, opt = self.opt, assert_cube=False)

        self.camencode = ImgCamEncode(img_args['chain_channels'], self.downsample)

        # 用于图像projection
        self.discrete_ratio = args['fusion_args']['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['fusion_args']['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]

        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'], num_point_features=4, voxel_size=args['voxel_size'], point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.compression = False
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(256, args['anchor_number'], kernel_size=1)    # 分类头, True/False 256 -> 2
        self.reg_head = nn.Conv2d(256, 7 * args['anchor_number'], kernel_size=1)    # 回归头, x/y/z/h/w/l/yaw 256, 14
        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # voxel features shape: (max_voxels=N, max_points_per_voxel=32, (x,y,z,intensity)=4)
        #print('voxel_features: ', batch_dict['voxel_features'].shape)
        # n, 4 -> n, c

        batch_dict = self.pillar_vfe(batch_dict)
        #print('pillar_features: ', batch_dict['pillar_features'].shape)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        #print('scatter_features: ', batch_dict['spatial_features'].shape)
        batch_dict = self.backbone(batch_dict)
        #print('backbone_features: ', batch_dict['spatial_features_2d'].shape)

        # 处理图像分支
        # process image to get bev        
        image_inputs_dict = data_dict['image_inputs']
        x, intrins, extrins = image_inputs_dict['imgs'], image_inputs_dict['intrins'], image_inputs_dict['extrins']
        B, N, C, imH, imW = x.shape     # torch.Size([4, N, 3, 320, 480])

        __p = lambda x: basic.pack_seqdim(x, B)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, B)

        features = self.camencode(x, record_len)
        features = self.feature2vox_simple(features, __p(intrins), __p(extrins), __p, __u)
        features = features.contiguous().view(B, -1, self.Y, self.X)
        if self.proj_first:
            features = self.proj_img_feat(features, data_dict['img_pairwise_t_matrix'], N_cams=N, record_len=record_len)

        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        #print('downsample_features: ', spatial_features_2d.shape)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        #print('compression_features: ', spatial_features_2d.shape)
        # dcn
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        #print('dcn_features: ', spatial_features_2d.shape)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        #print(psm_single.shape, rm_single.shape)

        # print('spatial_features_2d: ', spatial_features_2d.shape)
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                features,
                batch_dict['spatial_features'],
                psm_single,
                record_len,
                pairwise_t_matrix, 
                self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head]
            )
            #print('fused_feature: ', fused_feature.shape, communication_rates)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                features,
                spatial_features_2d,
                psm_single,
                record_len,
                pairwise_t_matrix
            )
            
        #print('fused_feature: ', fused_feature.shape)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm, 'rm': rm}
        output_dict.update(result_dict)
        output_dict.update({
            'mask': 0,
            'comm_rate': communication_rates
        })
        
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []

        #print('record_len: ', record_len)
        #print(psm_single.shape)
        #for nn in split_psm_single:
        #    print(nn.shape)
        
        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)
        output_dict.update({'psm_single_v': psm_single_v,
                       'psm_single_i': psm_single_i,
                       'rm_single_v': rm_single_v,
                       'rm_single_i': rm_single_i,
                       'comm_rate': communication_rates
                       })
        return output_dict

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

    def proj_img_feat(self, img_feat, pairwise_t_matrix, N_cams, record_len):
        _, C, H, W = img_feat.shape
        B, L = pairwise_t_matrix.shape[:2]

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        batch_img_feats = self.regroup(img_feat, record_len)

        feat_fuse = []
        for b in range(B):
            # number of cameras
            N = record_len[b]
            # (N,N,4,4). t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            batch_img_feat = batch_img_feats[b]
            # C, H, W = batch_img_feat.shape[1:]
            batch_img_feat = warp_affine_simple(batch_img_feat, t_matrix[0, :, :, :], (H, W))
            feat_fuse.append(batch_img_feat)

        feat_fuse = torch.concat(feat_fuse, dim=0)
        return feat_fuse

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
