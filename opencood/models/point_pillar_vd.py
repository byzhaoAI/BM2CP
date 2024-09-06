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

from opencood.utils.common_utils import torch_tensor_to_numpy

from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor

from opencood.models.vd_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.vd_modules.attentioncomm import AttenComm

from opencood.models.vd_modules.sensor_blocks import ImgCamEncode
from opencood.models.vd_modules.camera_encode_blocks import CamEncode
from opencood.models.vd_modules.utils import basic, vox, geom
from opencood.models.vd_modules.autoencoder_conv import Autoencoder

from opencood.models.vd_modules.swap_fusion_modules import SwapFusionEncoder


def _regroup(dense_feature, record_len, max_len):
    """
    Regroup the data based on the record_len.
    Parameters
    ----------
    dense_feature : torch.Tensor
        N, M, C, H, W
    record_len : list
        [sample1_len, sample2_len, ...]
    max_len : int
        Maximum cav number
    Returns
    -------
    regroup_feature : torch.Tensor
        B, L, C, H, W
    """
    cum_sum_len = list(np.cumsum(torch_tensor_to_numpy(record_len)))
    split_features = torch.tensor_split(dense_feature, cum_sum_len[:-1])
    
    regroup_features = []
    mask = []

    for split_feature in split_features:
        # L, M, C, H, W
        feature_shape = split_feature.shape

        # the maximum L is 5 as most 5 cavs
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)

        padding_tensor = torch.zeros(padding_len, feature_shape[1], feature_shape[2], feature_shape[3], feature_shape[4])
        padding_tensor = padding_tensor.to(split_feature.device)

        # max_len, M, C, H, W  
        split_feature = torch.cat([split_feature, padding_tensor], dim=0)
        
        # 1, 5C, H, W
        # split_feature = split_feature.view(-1, feature_shape[2], feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    # B, 5C, H, W
    # regroup_features = torch.cat(regroup_features, dim=0)
    # B, L, C, H, W
    # regroup_features = rearrange(regroup_features, 'b (l c) h w -> b l c h w', l=max_len)

    # B, max_len, M, C, H, W
    regroup_features = torch.stack(regroup_features, dim=0)
    # B, max_len
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)

    return regroup_features, mask


class MultiModalFusion(nn.Module):
    def __init__(self, num_modality, dim, threshold=0.1):
        super().__init__()
        self.threshold = threshold
        self.autoencoder = Autoencoder()
        self.rec_loss = nn.MSELoss()

    def forward(self, feats, modality_adapter):
        # 模态融合 img, pc, radar: B*C*Y*X
        auto_enc_loss = 0
        B, C, H, W = feats[0].shape

        main_counts = []
        for idx, feat in enumerate(feats):
            feat_mid, feat_rec = self.autoencoder(feat)
            feat_svd = feat_mid.flatten(2)
            # b*c*c, b*c, b*n*n
            feat_s, feat_v, feat_d = torch.linalg.svd(feat_svd.cpu())

            # 按通道统计大于阈值的元素个数
            counts = torch.sum(feat_v > self.threshold, dim=1)
            main_counts.append(counts)
            auto_enc_loss = auto_enc_loss + self.rec_loss(feat, feat_rec)
        # B,1
        domain_idx = torch.argmax(torch.stack(main_counts, dim=1), dim=1).unsqueeze(1)
        # B,M,C,H,W
        con_feat = torch.stack(feats, dim=1)
        # 创建一个包含批量索引的张量
        batch_indices = torch.arange(B).unsqueeze(1)
        # 使用高级索引提取对应的特征
        domain_features = con_feat[batch_indices, domain_idx, :, :, :].repeat(1,len(feats),1,1,1)

        weights = torch.einsum('smchw, snchw -> smnhw', (domain_features, con_feat))
        weights = F.softmax(weights, 2)
        weights = torch.einsum('smnhw, snchw -> smchw', (weights, con_feat))
        weights = F.softmax(weights, 1)
        
        fused_feat = torch.sum(con_feat * weights, dim=1)
        return fused_feat, auto_enc_loss


class PointPillarVD(nn.Module):  
    def __init__(self, args):
        super(PointPillarVD, self).__init__()
        # cuda选择
        self.use_radar = args['use_radar']
        self.max_cav = args['max_cav']
        self.device = args['device'] if 'device' in args else 'cpu'
        self.supervise_single = args['supervise_single'] if 'supervise_single' in args else False
        
        # camera 分支网络
        img_args = args['img_params']
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
        print("Number of parameter CamEncode: %d" % (sum([param.nelement() for param in self.camencode.parameters()])))

        # lidar 分支网络
        pc_args = args['pc_params']
        self.pillar_vfe = PillarVFE(pc_args['pillar_vfe'], num_point_features=4, voxel_size=pc_args['voxel_size'], point_cloud_range=pc_args['lidar_range'])
        print("Number of parameter pillar_vfe: %d" % (sum([param.nelement() for param in self.pillar_vfe.parameters()])))
        self.scatter = PointPillarScatter(pc_args['point_pillar_scatter'])
        print("Number of parameter scatter: %d" % (sum([param.nelement() for param in self.scatter.parameters()])))

        # radar 分支网络
        self.radar_enc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 双模态融合
        modality_args = args['modality_fusion']
        self.modal_multi_scale = modality_args['bev_backbone']['multi_scale']
        self.num_levels = len(modality_args['bev_backbone']['layer_nums'])
        assert img_args['bev_dim'] == pc_args['point_pillar_scatter']['num_features']
        self.modality_adapter = nn.Embedding(modality_args['num_modality'], img_args['bev_dim'])
        self.fusion = MultiModalFusion(modality_args['num_modality'], img_args['bev_dim'])
        print("Number of parameter modal fusion: %d" % (sum([param.nelement() for param in self.fusion.parameters()])))
        self.backbone = ResNetBEVBackbone(modality_args['bev_backbone'], input_channels=pc_args['point_pillar_scatter']['num_features'])
        print("Number of parameter bevbackbone: %d" % (sum([param.nelement() for param in self.backbone.parameters()])))

        self.shrink_flag = False
        if 'shrink_header' in modality_args:
            self.shrink_flag = modality_args['shrink_header']['use']
            self.shrink_conv = DownsampleConv(modality_args['shrink_header'])
            print("Number of parameter shrink_conv: %d" % (sum([param.nelement() for param in self.shrink_conv.parameters()])))
        
        self.compression = False
        if 'compression' in modality_args and modality_args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, modality_args['compression'])
        map_channels = input_dim//compress_raito if self.compression else modality_args['shrink_header']['dim'][0] if self.shrink_flag else sum(modality_args['bev_backbone']['num_upsample_filter'])

        # 协作融合网络
        # self.multi_scale = args['collaborative_fusion']['multi_scale']
        # self.fusion_net = AttenComm(args['collaborative_fusion'])
        self.fusion_net = SwapFusionEncoder(args['fax_fusion'])
        print("Number of fusion_net parameter: %d" % (sum([param.nelement() for param in self.fusion_net.parameters()])))
        self.fusion_cls_head = []

        # 预测头
        self.outC = args['outC']
        self.cls_head = nn.Conv2d(self.outC, args['anchor_number'], kernel_size=1)               
        self.reg_head = nn.Conv2d(self.outC, 7 * args['anchor_number'], kernel_size=1)
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.outC, args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) # BIN_NUM = 2
        else:
            self.use_dir = False

        # freeze
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

        for p in self.bevbackbone.parameters():
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

    def mask_modality(self, x, y, adapter, record_len):
        """
        # 1. C+L / L and L / C+L
        split_x = self.regroup(x, record_len)
        new_x = []
        for _x in split_x:
            B, L, C, H, W = _x.shape
            
            if B > 1:
                new_x.append(torch.cat([_x[0].unsqueeze(0), torch.zeros(B-1, L, C, H, W).to(_x.device)]))
            else:
                new_x.append(_x)
            
            if B > 1:
                _other = _x[1:]
                if _other.dim() == 4:
                    _other = _other.unsqueeze(0)
                new_x.append(torch.cat([torch.zeros(1, L, C, H, W).to(_x.device), _other]))
            else:
                new_x.append(torch.zeros(_x.shape).to(_x.device))
            
        new_x = torch.cat(new_x)
        new_y = y
        new_adapter = adapter
        
        # 2. C+L / C and C / C+L
        split_y = self.regroup(y, record_len)
        new_y = []
        for _y in split_y:
            B, C, H, W = _y.shape
            
            if B > 1:    
                new_y.append(torch.cat([_y[0].unsqueeze(0), torch.zeros(B-1, C, H, W).to(_y.device)]))
            else:
                new_y.append(_y)
            
            if B > 1:
                _other = _y[1:]
                if _other.dim() == 3:
                    _other = _other.unsqueeze(0)
                new_y.append(torch.cat([torch.zeros(1, C, H, W).to(_y.device), _other]))
            else:
                new_y.append(torch.zeros(_y.shape).to(_y.device))
            
        new_y = torch.cat(new_y)
        new_x = x
        new_adapter = adapter
        
        """
        # 3. L/L and C/C
        new_x = torch.zeros(x.shape).to(x.device)
        new_y = y
        """
        new_x = x
        new_y = torch.zeros(y.shape).to(y.device)
        
        # 4. L/C and C/L
        split_x = self.regroup(x, record_len)
        split_y = self.regroup(y, record_len)
        new_x, new_y = [], []
        for _x, _y in zip(split_x, split_y):
            _B, L, _, _H, _W = _x.shape
            B, C, H, W = _y.shape
            assert _B == B
            
            # C + L
            if B > 1:
                new_x.append(torch.cat([_x[0].unsqueeze(0), torch.zeros(B-1, L, 3, _H, _W).to(_x.device)]))
                _other = _y[1:]
                if _other.dim() == 3:
                    _other = _other.unsqueeze(0)
                new_y.append(torch.cat([torch.zeros(1, C, H, W).to(_y.device), _other]))
            else:
                new_x.append(_x)
                new_y.append(torch.zeros(_y.shape).to(_y.device))
            
            # L + C
            if B > 1:
                new_y.append(torch.cat([_y[0].unsqueeze(0), torch.zeros(B-1, C, H, W).to(_y.device)]))
                _other = _x[1:]
                if _other.dim() == 4:
                    _other = _other.unsqueeze(0)
                new_x.append(torch.cat([torch.zeros(1, L, 3, _H, _W).to(_x.device), _other]))
            else:
                new_y.append(_y)
                new_x.append(torch.zeros(_x.shape).to(_x.device))
            
        new_x, new_y = torch.cat(new_x), torch.cat(new_y)
        """
        #return new_x, new_y, adapter
        return x, y, adapter


    def forward(self, data_dict, mode=[0,1,2]):   # loss: 5.91->0.76
        # get two types data
        image_inputs_dict = data_dict['image_inputs']
        pc_inputs_dict = data_dict['processed_lidar']
        if 'processed_radar' in data_dict:
            radar_feature = data_dict['processed_radar']
        else:
            radar_feature = None
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        # process lidar point cloud
        batch_dict = {'voxel_features': pc_inputs_dict['voxel_features'],
                      'voxel_coords': pc_inputs_dict['voxel_coords'],
                      'voxel_num_points': pc_inputs_dict['voxel_num_points'],
                      'record_len': record_len}
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        lidar_feature = batch_dict['spatial_features']
        # batch_dict = self.backbone(batch_dict)
        # spatial_features_2d = batch_dict['spatial_features_2d'] 
        
        # process RGB image
        x, intrins, extrins = image_inputs_dict['imgs'], image_inputs_dict['intrins'], image_inputs_dict['extrins']
        B, N, C, imH, imW = x.shape     # torch.Size([4, N, 3, 320, 480])

        __p = lambda x: basic.pack_seqdim(x, B)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, B)

        img_feature = self.camencode(x)
        img_feature = rearrange(img_feature, 'b n c h w -> (b n) c h w')
        img_feature = self.feature2vox_simple(img_feature, __p(intrins), __p(extrins), __p, __u)
        img_feature = rearrange(img_feature, 'b c h w z-> b (c z) h w')

        modal_features = []
        if 0 in mode:
            modal_features.append(lidar_feature)
        if 1 in mode:
            modal_features.append(img_feature)

        # if radar_feature is not None:
        if self.use_radar:
            # process radar point cloud
            radar_feature = rearrange(radar_feature, 'b c z y x -> b (c z) y x')
            radar_feature = self.radar_enc(radar_feature)
            
            if 2 in mode:
                modal_features.append(radar_feature)

        # x = self.fusion([lidar_feature], self.modality_adapter)
        x, rec_loss = self.fusion(modal_features, self.modality_adapter)

        # b, m, c, h, w
        modal_features.append(x)
        x = torch.stack(modal_features, dim=1)   

        x_B, x_M, x_C, x_H, x_W = x.shape
        x = rearrange(x, 'b m c h w -> (b m) c h w')
        spatial_features_2d = self.backbone({'spatial_features': x})['spatial_features_2d']
        
        if self.shrink_flag:    # downsample feature to reduce memory
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:    # compressor
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        
        spatial_features_2d = rearrange(spatial_features_2d, '(b m) c h w -> b m c h w', b=x_B, m=x_M)
        # regroup_feature: (B, max_len, M, C, H, W)
        # mask: (B, max_len)
        regroup_feature, mask = _regroup(spatial_features_2d, record_len, self.max_cav)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        com_mask = repeat(com_mask, 'b m h w c l -> b (m new_m) (h new_h) (w new_w) c l',
                        new_m=x_M,
                        new_h=regroup_feature.shape[4],
                        new_w=regroup_feature.shape[5])

        regroup_feature = rearrange(regroup_feature, 'b l m c h w -> (b m) l c h w')
        com_mask = rearrange(com_mask, 'b m h w c l -> (b m) h w c l')
        fused_feature = self.fusion_net(regroup_feature, com_mask)
        
        # decode head
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        psm = rearrange(psm, '(b m) c h w -> b m c h w', b=mask.shape[0], m=x_M)
        rm = rearrange(rm, '(b m) c h w -> b m c h w', b=mask.shape[0], m=x_M)
        # update output dict
        output_dict = {
            'psm': psm[:,-1], 
            'rm': rm[:,-1], 
            'rec_loss': rec_loss,
            'modality_num': x_M-1,
        }

        for x_idx in range(x_M-1):
            output_dict.update({
                'psm_{}'.format(x_idx): psm[:,x_idx],
                'rm_{}'.format(x_idx): rm[:,x_idx],
            })


        output_dict.update({
            'cls_preds': psm,
            'reg_preds': rm,
            # 'dir_preds': dir_preds
        })

        if self.use_dir:
            dm = self.dir_head(fused_feature)
            dm = rearrange(dm, '(b m) c h w -> b m c h w', b=mask.shape[0], m=x_M)
            output_dict.update({"dm": dm[:,-1]})
            for x_idx in range(x_M-1):
                output_dict.update({
                    'dm_{}'.format(x_idx): dm[:,x_idx],
                })

        # output_dict.update({
        #     'comm_rate': communication_rates
        # })

        if not self.supervise_single:
            return output_dict

        # single decode head
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        
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
                       'mask': mask,
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
