"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler

Intermediate fusion for camera based collaboration
"""
import numpy as np
from einops import rearrange
import time

import torch
from torch import nn
import torch.nn.functional as F

from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone as PCBaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor

from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization

from opencood.models.realcp_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.realcp_modules.attentioncomm_corr_newattnfuse_mamba import AttenComm
from opencood.models.realcp_modules.sensor_blocks import ImgCamEncode


class MultiModalFusion(nn.Module):
    def __init__(self, num_modality, dim):
        super().__init__()
        self.adapt_conv = nn.Conv2d(dim*num_modality, dim, kernel_size=1)
        self.mlp = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, feats, modality_adapter):
        
        fused_feat_list = []
        for i in range(len(feats)):
            embed = modality_adapter.weight[i].unsqueeze(-1).unsqueeze(-1)
            fused_feat_list.append(embed * feats[i])
        fused_feat = torch.concat(fused_feat_list, dim=1)
        fused_feat = self.adapt_conv(fused_feat)
        
        B, C, H, W = fused_feat.shape
        fused_feat = rearrange(fused_feat, 'b c h w -> (h w) b c')
        retrived_feat = 0
        for feat in feats:
            score = torch.bmm(rearrange(feat, 'b c h w -> (h w) b c'), fused_feat.transpose(1, 2)) / np.sqrt(C)
            attn = F.softmax(score, -1)
            retrived_feat = retrived_feat + torch.bmm(attn, fused_feat)
        fused_feat = fused_feat + self.relu(self.mlp(retrived_feat))
        fused_feat = rearrange(fused_feat, '(h w) b c -> b c h w', h=H, w=W)
        
        """
        stack_feat = torch.stack(feats, dim=1)
        embeds = modality_adapter.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        fused_feat = rearrange(stack_feat*embeds, 'b m c h w -> b (m c) h w')
        fused_feat = self.adapt_conv(fused_feat)
        
        _, _, C, H, W = stack_feat.shape
        stack_feat = rearrange(stack_feat, 'b m c h w -> m (h w) b c')
        fused_feat = (rearrange(fused_feat, 'b c h w -> (h w) b c')).unsqueeze(0)
        # m S b c @ 1 S c b -> m S b b
        attn = torch.matmul(stack_feat, fused_feat.transpose(2, 3)) / np.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        # m S b b @ 1 S b c -> m S b c
        retrived_feat = torch.matmul(attn, fused_feat)
        retrived_feat = torch.sum(retrived_feat, dim=0, keepdim=False)
        
        # S b c + S b c -> S b c
        fused_feat = fused_feat.squeeze(0) + F.relu(self.mlp(retrived_feat))
        fused_feat = rearrange(fused_feat, '(h w) b c -> b c h w', h=H, w=W)
        """
        return fused_feat

    
class PointPillarRealCPMambaNewAttnFuse(nn.Module):
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 原始图片大小  ogfH:128  ogfW:288
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 下采样16倍后图像大小  fH: 12  fW: 22
        # ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(41x12x22)
        ds = torch.tensor(depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode']), dtype=torch.float).view(-1,1,1).expand(-1, fH, fW)

        D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 在0到288上划分18个格子 xs: DxfHxfW(41x12x22)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 在0到127上划分8个格子 ys: DxfHxfW(41x12x22)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return frustum
    
    def __init__(self, args):
        super(PointPillarRealCPMambaNewAttnFuse, self).__init__()
        # cuda选择
        self.device = args['device'] if 'device' in args else 'cpu'
        self.supervise_single = args['supervise_single'] if 'supervise_single' in args else False
        
        # camera 分支网络
        img_args = args['img_params']
        self.grid_conf = img_args['grid_conf']   # 网格配置参数
        self.data_aug_conf = img_args['data_aug_conf']   # 数据增强配置参数
        self.downsample = img_args['img_downsample']  # 下采样倍数
        self.bevC = img_args['bev_dim']  # 图像特征维度
        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        
        # 用于投影到BEV视角的参数
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'],)  # 划分网格
        self.dx = dx.clone().detach().requires_grad_(False).to(torch.device(self.device))  # [0.4,0.4,20]
        self.bx = bx.clone().detach().requires_grad_(False).to(torch.device(self.device))  # [-49.8,-49.8,0]
        self.nx = nx.clone().detach().requires_grad_(False).to(torch.device(self.device))  # [250,250,1]
        self.frustum = self.create_frustum().clone().detach().requires_grad_(False).to(torch.device(self.device))  # frustum: DxfHxfWx3
        self.D, _, _, _ = self.frustum.shape
        print('total depth levels: ', self.D)
        self.camencode = ImgCamEncode(self.D, self.bevC, self.downsample, self.grid_conf['ddiscr'], self.grid_conf['mode'], img_args['use_depth_gt'], img_args['depth_supervision'])
        print("Number of parameter CamEncode: %d" % (sum([param.nelement() for param in self.camencode.parameters()])))
        
        # lidar 分支网络
        pc_args = args['pc_params']
        self.pillar_vfe = PillarVFE(pc_args['pillar_vfe'], num_point_features=4, voxel_size=pc_args['voxel_size'], point_cloud_range=pc_args['lidar_range'])
        print("Number of parameter pillar_vfe: %d" % (sum([param.nelement() for param in self.pillar_vfe.parameters()])))
        self.scatter = PointPillarScatter(pc_args['point_pillar_scatter'])
        print("Number of parameter scatter: %d" % (sum([param.nelement() for param in self.scatter.parameters()])))
        
        # 双模态融合
        modality_args = args['modality_fusion']
        self.modal_multi_scale = modality_args['bev_backbone']['multi_scale']
        self.num_levels = len(modality_args['bev_backbone']['layer_nums'])
        assert img_args['bev_dim'] == pc_args['point_pillar_scatter']['num_features']
        self.modality_adapter = nn.Embedding(modality_args['num_modality'], img_args['bev_dim'])
        self.fusion = MultiModalFusion(modality_args['num_modality'], img_args['bev_dim'])
        # self.fusion = MultiModalFusion_Mamba(modality_args['num_modality'], img_args['bev_dim'])
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
        self.multi_scale = args['collaborative_fusion']['multi_scale']
        self.fusion_net = AttenComm(args['collaborative_fusion'])
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
        

        # 3. L/L and C/C
        new_x = torch.zeros(x.shape).to(x.device)
        new_y = y

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


    def forward(self, data_dict):   # loss: 5.91->0.76
        # get two types data
        image_inputs_dict = data_dict['image_inputs']
        pc_inputs_dict = data_dict['processed_lidar']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        # process point cloud 单分支时的网络部分包含4部分，双分支只包含前两部分
        #（1）PillarVFE              pcdet/models/backbones_3d/vfe/pillar_vfe.py   # 3D卷积, 点特征编码
        #（2）PointPillarScatter     pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py   # 2D卷积，创建（实际就是变形）一个大小为(C，H，W)的伪图像
        #（3）BaseBEVBackbone        pcdet/models/backbones_2d/base_bev_backbone.py   # 2D卷积
        #（4）AnchorHeadSingle       pcdet/models/dense_heads/anchor_head_single.py   # 检测头
        batch_dict = {'voxel_features': pc_inputs_dict['voxel_features'],
                      'voxel_coords': pc_inputs_dict['voxel_coords'],
                      'voxel_num_points': pc_inputs_dict['voxel_num_points'],
                      'record_len': record_len}
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        # batch_dict = self.backbone(batch_dict)
        # spatial_features_2d = batch_dict['spatial_features_2d'] 
        
        # process image to get bev
        # x, rots, trans, intrins, post_rots, post_trans, depth_map = image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans'], image_inputs_dict['depth_map']
        # geom: ([8, 1, 48, 40, 60, 3]), x: torch.Size([8, 1, 48, 40, 60, 64])
        # geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3
        geom = self.get_geometry(image_inputs_dict)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3
        # get_cam_feats, 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64) Return B x N x D x H/downsample x W/downsample x C
        x = image_inputs_dict['imgs']

        x, spatial_features, modality_adapter = self.mask_modality(x, batch_dict['spatial_features'], self.modality_adapter, record_len)
        
        B, N, C, imH, imW = x.shape     # torch.Size([4, 1, 3, 320, 480])
        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x:  B: 4  N: 4  C: 3  imH: 256  imW: 352 -> 16 x 4 x 256 x 352
        _, x = self.camencode(x)     # x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22) -> 多了一个维度D：代表深度
        x = x.view(B, N, self.bevC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)
        # 将图像转换到voxel
        x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 240 x 240
        # 转换到BEV下, collapse Z
        x = torch.cat(x.unbind(dim=2), 1)  # 消除掉z维
        
        # voxel下的模态融合 img: B*C*Z*Y*X; pc: B*C*Z*Y*X
        #start_time = time.time()
        x = self.fusion([x, spatial_features], modality_adapter)
        #print('modality fusion time: ', time.time() - start_time)
        
        #x = self.fusion([x, batch_dict['spatial_features']], self.modality_adapter)
        batch_dict['spatial_features'] = x
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:    # downsample feature to reduce memory
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:    # compressor
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # collaborative fusion
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                                            batch_dict['spatial_features'],
                                            self.cls_head(spatial_features_2d),
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone,
                                            [self.shrink_conv, self.cls_head, self.reg_head])
            #print('fused_feature: ', fused_feature.shape, communication_rates)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                                            spatial_features_2d,
                                            self.cls_head(spatial_features_2d),
                                            record_len,
                                            pairwise_t_matrix)
        
        # decode head
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        # update output dict
        output_dict = {'psm': psm, 'rm': rm}

        if self.use_dir:
            dm = self.dir_head(fused_feature)
            output_dict.update({"dm": dm})

        output_dict.update({
            'comm_rate': communication_rates
        })

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

    #def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
    def get_geometry(self, image_inputs_dict):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
                # process image to get bev
        rots, trans, intrins, post_rots, post_trans = image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']

        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目) DAIR数据集只有1个相机

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if post_rots.device != 'cpu':
            inv_post_rots = torch.inverse(post_rots.to('cpu')).to(post_rots.device)
        else:
            inv_post_rots = torch.inverse(post_rots)
        points = inv_post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        #print(intrins.shape)
        if intrins.device != 'cpu':
            inv_intrins = torch.inverse(intrins.to('cpu')).to(intrins.device)
        else:
            inv_intrins = torch.inverse(intrins)
        
        combine = rots.matmul(inv_intrins)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]

        return points  # B x N x D x H x W x 3 (4 x 1 x 41 x 16 x 22 x 3)

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        # collapsed_final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        # return collapsed_final#, x  # final: 4 x 64 x 240 x 240  # B, C, H, W
        return final
