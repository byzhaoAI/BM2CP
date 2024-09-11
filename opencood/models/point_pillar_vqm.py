""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone as PCBaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor

from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization

from opencood.models.vqm_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.realcp_modules.sensor_blocks import ImgCamEncode
from opencood.models.vqm_modules.pyramid_fuse import PyramidFusion
from opencood.models.vqm_modules.autoencoder_conv import Autoencoder

import torchvision


def normalize_pairwise_tfm(pairwise_t_matrix, H, W, discrete_ratio, downsample_rate=1):
    """
    normalize the pairwise transformation matrix to affine matrix need by torch.nn.functional.affine_grid()
    Args:
        pairwise_t_matrix: torch.tensor
            [B, L, L, 4, 4], B batchsize, L max_cav
        H: num.
            Feature map height
        W: num.
            Feature map width
        discrete_ratio * downsample_rate: num.
            One pixel on the feature map corresponds to the actual physical distance

    Returns:
        affine_matrix: torch.tensor
            [B, L, L, 2, 3]
    """

    affine_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
    affine_matrix[...,0,1] = affine_matrix[...,0,1] * H / W
    affine_matrix[...,1,0] = affine_matrix[...,1,0] * W / H
    affine_matrix[...,0,2] = affine_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
    affine_matrix[...,1,2] = affine_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2

    return affine_matrix


class MultiModalFusion(nn.Module):
    def __init__(self, num_modality, dim, ratio=0.8, threshold=0.1):
        super().__init__()
        self.dim =  256
        self.threshold = threshold
        self.ratio = ratio
        self.autoencoder = Autoencoder()
        self.rec_loss = nn.MSELoss()
        self.value_func = nn.Sequential(
            nn.Conv2d(dim * 2, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Sigmoid(),
        )

        self.s_func = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Sigmoid(),
        )

        self.v_func = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Sigmoid(),
        )

        self.d_func = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Sigmoid(),
        )

    def forward(self, feats, training):
        # 模态融合 img, pc, radar: B*C*Y*X

        con_feat = torch.stack(feats, dim=1)
        B, M, C, H, W = con_feat.shape
        con_feat = rearrange(con_feat, 'b m c h w -> (b m) c h w')
        
        feat_s = self.s_func(con_feat)
        feat_v = self.v_func(con_feat)
        feat_d = self.d_func(con_feat)
        print(feat_s.shape, feat_v.shape, feat_d.shape)
        print(aaa)
        feat_mid, auto_enc_loss = self.cal_rec_loss(con_feat, training)

        # 按通道统计大于阈值的元素个数
        feat_v = rearrange(feat_v, '(b m) c -> b m c', b=B, m=M)
        feat_s, feat_d = map(lambda x: rearrange(x, '(b m) h w -> b m h w', b=B, m=M), (feat_s, feat_d))

        # b*c*c, b*c, b*n*n
        new_s, new_d = map(lambda x: torch.zeros((B, *x.shape[2:])).to(feat_svd.device), (feat_s, feat_d))
        new_v = torch.zeros((B, new_s.shape[1], new_d.shape[1])).to(feat_svd.device)
        
        counts = torch.sum(feat_v > self.threshold, dim=-1)
        total_counts = torch.sum(counts, dim=-1)
        for b in range(B):
            count = counts[b]
            if total_counts[b] > min(new_v.shape[1], new_v.shape[2]):
                ratio = count / total_counts[b]
                count = torch.round(min(new_v.shape[1], new_v.shape[2]) * ratio).to(torch.int)
                if M > 1:
                    count[-1] = self.dim - torch.sum(count[:-1])
            # print(count)

            index = 0
            for _m in range(M):
                # (b*m)*c*c, (b*m)*c, (b*m)*n*n
                # b*c*c, b*c, b*n*n
                # print(new_s[b,index:index+count[_m],:].shape, feat_s[b,_m,:count[_m],:].shape)
                new_s[b,index:index+count[_m],:] = feat_s[b,_m,:count[_m],:]
                new_d[b,:,index:index+count[_m]] = feat_d[b,_m,:,:count[_m]]

                # 创建索引张量, 生成从 a 到 b 的索引
                indices = torch.arange(index, min(index+count[_m], new_v.shape[1], new_v.shape[2]))
                # print('indices: ', indices, index, index+count[_m], new_v.shape[1], new_v.shape[2])
                # print(new_v[b,indices,indices].shape, feat_v[b,_m,:count[_m]].shape)
                new_v[b,indices,indices] = feat_v[b,_m,:count[_m]]

                index = index + count[_m]
        
        rec_feat = torch.bmm(torch.bmm(new_s,new_v), new_d)
        rec_feat = rearrange(rec_feat, 'b c (h w) -> b c h w', h=feat_mid.shape[-2], w=feat_mid.shape[-1])
        rec_feat = self.autoencoder.decoder(rec_feat)
        # -> [-1, 1]
        rec_feat = (rec_feat - rec_feat.min()) / (rec_feat.max() - rec_feat.min()) * 2 - 1.0

        scores = []
        for feat in feats:
            scores.append(self.value_func(torch.cat([rec_feat, feat], dim=1)))
        scores = torch.cat(scores, dim=1)
        scores = F.softmax(scores, dim=1).unsqueeze(2)

        con_feat = rearrange(con_feat, '(b m) c h w -> b m c h w', b=B, m=M)
        fused_feat = torch.sum((1 - self.ratio) * scores * con_feat, dim=1) + rec_feat * self.ratio
        return fused_feat, auto_enc_loss

    def cal_rec_loss(self, feat, training):
        feat_mid, feat_rec = self.autoencoder(feat)

        if training:
            feat_svd = feat_mid.flatten(2)
            # (b*m)*c*c, (b*m)*c, (b*m)*n*n
            feat_s, feat_v, feat_d = torch.linalg.svd(feat_svd.cpu())
            feat_s, feat_v, feat_d = map(lambda x: x.to(feat_svd.device), (feat_s, feat_v, feat_d))

            return feat_mid, self.rec_loss(feat, feat_rec)#, self.rec_loss()
        return feat_mid, 0


class PointPillarVQM(nn.Module):
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
        super(PointPillarVQM, self).__init__()
        # cuda选择
        self.use_radar = args['use_radar']
        self.max_cav = args['max_cav']
        self.cav_range = args['lidar_range']
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
        self.fusion = MultiModalFusion(modality_args['num_modality'], img_args['bev_dim'])
        # self.modal_conv = nn.Conv2d(img_args['bev_dim'] * 2, img_args['bev_dim'], kernel_size=1)
        # self.modality_adapter = nn.Embedding(modality_args['num_modality'], img_args['bev_dim'])
        # self.fusion = MultiModalFusion(modality_args['num_modality'], img_args['bev_dim'])
        # self.fusion = MultiModalFusion_Mamba(modality_args['num_modality'], img_args['bev_dim'])
        # print("Number of parameter modal fusion: %d" % (sum([param.nelement() for param in self.fusion.parameters()])))
        self.backbone = ResNetBEVBackbone(modality_args['bev_backbone'], input_channels=pc_args['point_pillar_scatter']['num_features'])
        print("Number of parameter bevbackbone: %d" % (sum([param.nelement() for param in self.backbone.parameters()])))

        self.shrink_flag = False
        if 'shrink_header' in modality_args:
            self.shrink_flag = modality_args['shrink_header']['use']
            self.shrink_conv = DownsampleConv(modality_args['shrink_header'])
            print("Number of parameter shrink_conv: %d" % (sum([param.nelement() for param in self.shrink_conv.parameters()])))
        
        self.pyramid_backbone = PyramidFusion(args['collaborative_fusion'])
        print("Number of parameter pyramid_backbone: %d" % (sum([param.nelement() for param in self.pyramid_backbone.parameters()])))
        # print(self.pyramid_backbone)

        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) # BIN_NUM = 2


    def forward(self, data_dict, mode=[0,1], training=True):
        output_dict = {'pyramid': 'collab'}
        # get two types data
        image_inputs_dict = data_dict['image_inputs']
        pc_inputs_dict = data_dict['processed_lidar']
        if 'processed_radar' in data_dict:
            radar_feature = data_dict['processed_radar']
        else:
            radar_feature = None
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
        lidar_feature = batch_dict['spatial_features']
        # batch_dict = self.backbone(batch_dict)
        
        # process image to get bev
        # x, rots, trans, intrins, post_rots, post_trans, depth_map = image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans'], image_inputs_dict['depth_map']
        # geom: ([8, 1, 48, 40, 60, 3]), x: torch.Size([8, 1, 48, 40, 60, 64])
        # geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3
        geom = self.get_geometry(image_inputs_dict)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3
        # get_cam_feats, 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64) Return B x N x D x H/downsample x W/downsample x C
        x = image_inputs_dict['imgs']

        x, spatial_features = x, batch_dict['spatial_features']
        
        B, N, C, imH, imW = x.shape     # torch.Size([4, 1, 3, 320, 480])
        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x:  B: 4  N: 4  C: 3  imH: 256  imW: 352 -> 16 x 4 x 256 x 352
        _, x = self.camencode(x)     # x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22) -> 多了一个维度D：代表深度
        x = x.view(B, N, self.bevC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)
        # 将图像转换到voxel
        x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 240 x 240
        # 转换到BEV下, collapse Z
        x = torch.cat(x.unbind(dim=2), 1)  # 消除掉z维

        # heter_feature_2d = self.modal_conv(torch.cat([x, spatial_features], dim=1))
        # heter_feature_2d = batch_dict['spatial_features']
        modal_features = []
        m_len = 0
        if 0 in mode:
            lidar_feature = torch.tanh(lidar_feature)
            modal_features.append(lidar_feature)
            m_len += 1
        if 1 in mode:
            x = torch.tanh(x)
            modal_features.append(x)
            m_len += 1

        # if radar_feature is not None:
        if self.use_radar:
            # process radar point cloud
            radar_feature = rearrange(radar_feature, 'b c z y x -> b (c z) y x')
            radar_feature = self.radar_enc(radar_feature)
            radar_feature = torch.tanh(radar_feature)
            
            if 2 in mode:
                modal_features.append(radar_feature)
                m_len += 1

        # x = self.fusion([lidar_feature], self.modality_adapter)
        x, rec_loss = self.fusion(modal_features, training=training)

        # b, m, c, h, w
        if training:
            modal_features.append(x)
            x = torch.stack(modal_features, dim=1)
        else:
            x = x.unsqueeze(1)

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        
        
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                x,
                                                record_len, 
                                                affine_matrix, 
                                                None, 
                                                None
                                            )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)

        dir_preds = self.dir_head(fused_feature)

        # (b*m_len,c,h,w) -> (b,m_len,c,h,w)
        if training:
            cls_preds = rearrange(cls_preds, '(b m) c h w -> b m c h w', b=len(record_len), m=m_len+1)
            reg_preds = rearrange(reg_preds, '(b m) c h w -> b m c h w', b=len(record_len), m=m_len+1)
            dir_preds = rearrange(dir_preds, '(b m) c h w -> b m c h w', b=len(record_len), m=m_len+1)

            output_dict.update({
                'cls_preds': cls_preds[:,-1],
                'reg_preds': reg_preds[:,-1],
                'dir_preds': dir_preds[:,-1],
                'rec_loss': rec_loss,
                'modality_num': m_len,
                'psm': cls_preds[:,-1],
                'rm': reg_preds[:,-1],
            })
            
            # output_dict.update({'occ_single_list': 
            #                     occ_outputs})

            for x_idx in range(m_len):
                output_dict.update({
                    'cls_preds_{}'.format(x_idx): cls_preds[:,x_idx],
                    'reg_preds_{}'.format(x_idx): reg_preds[:,x_idx],
                    'dir_preds_{}'.format(x_idx): dir_preds[:,x_idx],
                    # 'occ_single_list_{}'.format(x_idx): eval(f"split_occ_outputs{x_idx}"),
                })

            return output_dict

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            'rec_loss': rec_loss,
            'modality_num': m_len,
            'psm': cls_preds,
            'rm': reg_preds,
        })
        return output_dict
        

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

