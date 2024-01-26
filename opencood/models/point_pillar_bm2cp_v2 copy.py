"""
Modified from: Runsheng Xu and Yue Hu
Authors: anonymous

Intermediate fusion for camera based collaboration
"""
import numpy as np
from numpy import record
import torch
from torch import nn
from torchvision.models.resnet import resnet18

from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone as PCBaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor

from opencood.utils import camera_utils
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization

from opencood.models.bm2cp_v2_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.bm2cp_v2_modules.attentioncomm import MultiModalAttenComm
from opencood.models.bm2cp_v2_modules.sensor_blocks import ImgCamEncode
from opencood.models.bm2cp_v2_modules.utils import basic, vox, geom


class PointPillarBM2CPV2(nn.Module):   
    def __init__(self, args):
        super(PointPillarBM2CPV2, self).__init__()
        # cuda选择
        self.supervise_single = args['supervise_single'] if 'supervise_single' in args else False
        
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
        print("Number of parameter CamEncode: %d" % (sum([param.nelement() for param in self.camencode.parameters()])))
        
        # lidar 分支网络
        pc_args = args['pc_params']
        self.pillar_vfe = PillarVFE(pc_args['pillar_vfe'], num_point_features=4, voxel_size=pc_args['voxel_size'], point_cloud_range=pc_args['lidar_range'])
        print("Number of parameter pillar_vfe: %d" % (sum([param.nelement() for param in self.pillar_vfe.parameters()])))
        self.scatter = PointPillarScatter(pc_args['point_pillar_scatter'])
        print("Number of parameter scatter: %d" % (sum([param.nelement() for param in self.scatter.parameters()])))
        
        # 特征提取网络
        self.backbone = ResNetBEVBackbone(args['bev_backbone'], input_channels=pc_args['point_pillar_scatter']['num_features'])
        print("Number of parameter backbone: %d" % (sum([param.nelement() for param in self.backbone.parameters()])))

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = args['shrink_header']['use']
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            print("Number of parameter shrink_conv: %d" % (sum([param.nelement() for param in self.shrink_conv.parameters()])))
        
        self.compression = False
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(args['shrink_header']['dim'], args['compression'])

        # 模态融合和协作融合网络
        self.multi_scale = args['fusion']['multi_scale']
        self.fusion_net = MultiModalAttenComm(args['fusion'])
        print("Number of fusion_net parameter: %d" % (sum([param.nelement() for param in self.fusion_net.parameters()])))

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

    def forward(self, data_dict):   # loss: 5.91->0.76
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        record_len = data_dict['record_len']
        image_inputs_dict = data_dict['image_inputs']
        pc_inputs_dict = data_dict['processed_lidar']

        batch_dict = {'voxel_features': pc_inputs_dict['voxel_features'],
                      'voxel_coords': pc_inputs_dict['voxel_coords'],
                      'voxel_num_points': pc_inputs_dict['voxel_num_points'],
                      'record_len': record_len}

        # 处理点云分支.
        # process point cloud
        #（1）PillarVFE              pcdet/models/backbones_3d/vfe/pillar_vfe.py
        #（2）PointPillarScatter     pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py
        #（3）BaseBEVBackbone        pcdet/models/backbones_2d/base_bev_backbone.py
        #（4）AnchorHeadSingle       pcdet/models/dense_heads/anchor_head_single.py
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        # pc non-zero volume: 2443591/25804800
        batch_dict = self.backbone(batch_dict)
        # spatial_features_2d = batch_dict['spatial_features_2d'] 
        
        # 处理图像分支
        # process image to get bev        
        x, intrins, extrins = image_inputs_dict['imgs'], image_inputs_dict['intrins'], image_inputs_dict['extrins']
        B, N, C, imH, imW = x.shape     # torch.Size([4, N, 3, 320, 480])

        __p = lambda x: basic.pack_seqdim(x, B)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, B)

        features = self.camencode(x, record_len)
        features = self.feature2vox_simple(features, __p(intrins), __p(extrins), __p, __u)
        features = features.contiguous().view(B, -1, self.Y, self.X)

        # Feature Compression
        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:    # downsample feature to reduce memory
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:    # compressor
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        
        # collaborative fusion        
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                features,
                batch_dict['spatial_features'],
                self.cls_head(spatial_features_2d),
                record_len,
                pairwise_t_matrix, 
                self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head]
            )
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                features,
                spatial_features_2d,
                self.cls_head(spatial_features_2d),
                record_len,
                pairwise_t_matrix
            )
        
        # decode head
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        
        # update output dict
        output_dict = {'psm': psm, 'rm': rm}

        if self.use_dir:
            dm = self.dir_head(fused_feature)
            output_dict.update({"dm": dm})

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
