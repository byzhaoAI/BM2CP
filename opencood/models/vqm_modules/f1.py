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

from opencood.models.realcp_modules.sensor_blocks import ImgCamEncode
from opencood.models.vqm_modules.layer_gnn import GraphConstructor, Mixprop
from opencood.models.vqm_modules.autoencoder_conv import Autoencoder



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
    def __init__(self, num_modality, dim, ratio=0.8, num_layers=3, threshold=0.1):
        super().__init__()
        self.dim =  256
        self.threshold = threshold
        self.ratio = ratio
        self.autoencoder = Autoencoder()
        self.rec_loss = nn.MSELoss()
        self.abs_loss = nn.L1Loss()

        self.value_func = nn.Sequential(
            nn.Conv2d(dim * 2, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Sigmoid(),
        )

        self.s_func = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((128, 128)),
        )

        self.v_func = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.d_func = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((256, 256)),
        )

        self.gc = GraphConstructor(nnodes=num_modality, dim=dim, alpha=3)
        self.num_layers = num_layers
        self.gconv1, self.gconv2, self.norm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.gconv1.append(Mixprop(dim, dim, gdep=2, dropout=0.3, alpha=0.05))
            self.gconv2.append(Mixprop(dim, dim, gdep=2, dropout=0.3, alpha=0.05))
            self.norm.append(nn.InstanceNorm2d(dim))

        self.skipE = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, feats, training):
        # 模态融合 img, pc, radar: B*C*Y*X

        con_feat = torch.stack(feats, dim=1)
        B, M, C, H, W = con_feat.shape
        con_feat = rearrange(con_feat, 'b m c h w -> (b m) c h w')

        feat_mid, feat_rec = self.autoencoder(con_feat)
        # b*c*c, b*c, b*n*n
        feat_v = self.v_func(feat_mid).flatten(1)
        
        auto_enc_loss, svd_loss = 0, 0
        if training:
            feat_s, feat_d = self.s_func(feat_mid).squeeze(1), self.d_func(feat_mid).squeeze(1)
            # construct diag matrix
            diag_v = torch.zeros((B*M, feat_s.shape[1], feat_d.shape[1])).to(con_feat.device)
            diag_v[:, :min(feat_s.shape[1], feat_d.shape[1]), :min(feat_s.shape[1], feat_d.shape[1])] = torch.diag_embed(feat_v)
            # recover matrix
            rec_feat_mid = torch.bmm(feat_s, torch.bmm(diag_v, feat_d))
            auto_enc_loss = self.rec_loss(con_feat, feat_rec)
            svd_loss = self.abs_loss(feat_mid.flatten(2), rec_feat_mid)

        # count principal components in channel dimension
        feat_v = rearrange(feat_v, '(b m) c -> b m c', b=B, m=M)
        counts = torch.sum(feat_v > self.threshold, dim=-1)
        best_indices = torch.argmax(counts, dim=-1)


        x = con_feat
        for i in range(self.num_layers):
            Adj = self.gc(x)
            residuals = self.gconv1[i](x, Adj) + self.gconv2[i](x, Adj.transpose(1,2))
            x = x + residuals
            x = x.view(B*M, C, H, W)
            x = self.norm[i](x)
            x = x.view(B, M, C, H, W)

        x = x.view(B*M, C, H, W)
        con_feat = con_feat.view(B*M, C, H, W)
        
        con_feat = con_feat + self.skipE(x)
        con_feat = F.relu(con_feat)
        con_feat = self.conv(con_feat)
        con_feat = con_feat.view(B, M, C, H, W)

        fused_feat = []
        for idx, index in enumerate(best_indices):
            fused_feat.append(con_feat[idx,index,:,:,:])
        return torch.stack(fused_feat, dim=0), auto_enc_loss, svd_loss


class CoVQMF1(nn.Module):   
    def __init__(self, args):
        super(CoVQMF1, self).__init__()
        # cuda选择
        # self.cav_range = args['lidar_range']
        self.device = args['device'] if 'device' in args else 'cpu'
        self.use_camera = args['use_camera'] if 'use_camera' in args else False
        self.use_lidar = args['use_lidar'] if 'use_lidar' in args else False
        assert self.use_lidar or self.use_camera, "at least use lidar or use camera."
        
        if self.use_camera:
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

        if self.use_lidar:
            # lidar 分支网络
            #（1）PillarVFE              pcdet/models/backbones_3d/vfe/pillar_vfe.py   # 3D卷积, 点特征编码
            #（2）PointPillarScatter     pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py   # 2D卷积，创建（实际就是变形）一个大小为(C，H，W)的伪图像
            pc_args = args['pc_params']
            self.pillar_vfe = PillarVFE(pc_args['pillar_vfe'], num_point_features=4, voxel_size=pc_args['voxel_size'], point_cloud_range=pc_args['lidar_range'])
            self.scatter = PointPillarScatter(pc_args['point_pillar_scatter'])

        if self.use_camera and self.use_lidar:
            # 双模态融合
            modality_args = args['modality_fusion']
            assert img_args['bev_dim'] == pc_args['point_pillar_scatter']['num_features']
            self.fusion = MultiModalFusion(modality_args['num_modality'], img_args['bev_dim'])
            self.camera_proj = nn.Conv2d(img_args['bev_dim'], img_args['bev_dim'], kernel_size=3, stride=1, padding=1)
            self.lidar_proj = nn.Conv2d(img_args['bev_dim'], img_args['bev_dim'], kernel_size=3, stride=1, padding=1)

    def minmax_norm(self, data):
        return (data - data.min()) / (data.max() - data.min()) * 2 - 1

    def forward(self, batch_dict, mode=[0,1], training=True):
        if self.use_lidar:
            batch_dict = self.pillar_vfe(batch_dict)
            batch_dict = self.scatter(batch_dict)
            lidar_feature = batch_dict['spatial_features']

            # lidar_feature = self.minmax_norm(lidar_feature)

        if self.use_camera:
            image_inputs_dict = batch_dict['image_inputs']
            geom = self.get_geometry(image_inputs_dict)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3
            x = image_inputs_dict['imgs']
            
            B, N, C, imH, imW = x.shape     # torch.Size([4, 1, 3, 320, 480])
            x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x:  B: 4  N: 4  C: 3  imH: 256  imW: 352 -> 16 x 4 x 256 x 352
            _, x = self.camencode(x)     # x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22) -> 多了一个维度D：代表深度
            x = x.view(B, N, self.bevC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
            x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)
            # 将图像转换到voxel
            x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 240 x 240
            # 转换到BEV下, collapse Z
            x = torch.cat(x.unbind(dim=2), 1)  # 消除掉z维

            # if not self.use_lidar:
            #     x = self.minmax_norm(x)

        # justify
        if self.use_lidar and self.use_camera:
            modal_features = []
            m_len = 0
            
            # process lidar to get bev
            if 0 in mode:
                modal_features.append(self.lidar_proj(lidar_feature))
                m_len += 1
            # process image to get bev
            if 1 in mode:  
                modal_features.append(self.camera_proj(x))
                m_len += 1
            # x = self.fusion([lidar_feature], self.modality_adapter)
            x, rec_loss, svd_loss = self.fusion(modal_features, training=training)

            # b, m, c, h, w
            if training:
                modal_features = [x] + modal_features
                x = torch.stack(modal_features, dim=1)
            else:
                x = x.unsqueeze(1)

            return x, m_len, rec_loss, svd_loss

        if self.use_lidar and not self.use_camera:
            return lidar_feature

        return x

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
