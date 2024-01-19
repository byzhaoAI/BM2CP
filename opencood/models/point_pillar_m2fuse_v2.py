"""
Modified from: Runsheng Xu and Yue Hu
Authors: anonymous

Intermediate fusion for camera based collaboration
"""

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
from opencood.models.m2fuse_v2_modules import cam_unproj

from opencood.models.where2comm_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.m2fuse_v2_modules.attentioncomm import ScaledDotProductAttention, AttenComm
from opencood.models.m2fuse_v2_modules.sensor_blocks import ImgCamEncode
#from opencood.models.m2fuse_v2_modal_interact_modules.attentioncomm import ScaledDotProductAttention, AttenComm
#from opencood.models.m2fuse_v2_modal_interact_modules.sensor_blocks import ImgCamEncode

# m2fuse_v2_module 和 m2fuse_v2_interact_nonselfatt_modules 是完全一样的

class ImgModalFusion(nn.Module):
    def __init__(self, dim, threshold=0.5):
        super().__init__()
        self.att = ScaledDotProductAttention(dim)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.Sigmoid()
        self.thres = threshold               

    def forward(self, img_voxel, pc_voxel):
        B, C, imZ, imH, imW = pc_voxel.shape
        pc_voxel = pc_voxel.view(B, C, -1)
        img_voxel = img_voxel.view(B, C, -1)
        voxel_mask = self.att(pc_voxel, img_voxel, img_voxel)
        voxel_mask = self.act(self.proj(voxel_mask.permute(0,2,1)))
        voxel_mask = voxel_mask.permute(0,2,1)
        voxel_mask = voxel_mask.view(B, C, imZ, imH, imW)

        ones_mask = torch.ones_like(voxel_mask).to(voxel_mask.device)
        zeros_mask = torch.zeros_like(voxel_mask).to(voxel_mask.device)
        mask = torch.where(voxel_mask>self.thres, ones_mask, zeros_mask)

        mask[0] = ones_mask[0]
        
        img_voxel = img_voxel.view(B, C, imZ, imH, imW)
        return img_voxel*mask


"""
class MultiModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.img_fusion = ImgModalFusion(dim)

        self.multigate = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.multifuse = nn.Conv3d(dim*2, dim, 1, 1, 0)

    def forward(self, img_voxel, pc_dict):
        pc_voxel = pc_dict['spatial_features_3d']
        B, C, Z, Y, X = pc_voxel.shape

        # pc->pc; img->img*mask; pc+img->
        ones_mask = torch.ones_like(pc_voxel).to(pc_voxel.device)
        zeros_mask = torch.zeros_like(pc_voxel).to(pc_voxel.device)
        
        pc_mask = torch.where(pc_voxel!=0, ones_mask, zeros_mask)
        pc_mask, _ = torch.max(pc_mask, dim=1)
        pc_mask = pc_mask.unsqueeze(1)
        img_mask = torch.where(img_voxel!=0, ones_mask, zeros_mask)
        img_mask, _ = torch.max(img_mask, dim=1)
        img_mask = img_mask.unsqueeze(1)

        fused_voxel = pc_mask*img_mask*self.multifuse(torch.cat([self.act(self.multigate(pc_voxel))*img_voxel, pc_voxel], dim=1))
        fused_voxel = fused_voxel + pc_voxel*pc_mask*(1-img_mask) + img_voxel*self.img_fusion(img_voxel, pc_voxel)*(1-pc_mask)*img_mask

        thres_map = pc_mask*img_mask*0 + pc_mask*(1-img_mask)*0.5 + (1-pc_mask)*img_mask*0.5 + (1-pc_mask)*(1-img_mask)*1
        # size = [B, 1, Z, Y, X]
        thres_map, _ = torch.min(thres_map, dim=2)  # collapse Z-axis, dim=4 size = [B, 1, Y, X]
        
        pc_dict['spatial_features'] = fused_voxel.view(B,C*Z, Y, X)
        return pc_dict, thres_map
"""


class MultiModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.img_fusion = ImgModalFusion(dim)

        self.multigate = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.multifuse = nn.Conv3d(dim*2, dim, 1, 1, 0)
        self.adapt_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.adapt_cls = nn.Sigmoid()

    def forward(self, img_voxel, pc_dict):
        pc_voxel = pc_dict['spatial_features_3d']
        B, C, Z, Y, X = pc_voxel.shape

        # pc->pc; img->img*mask; pc+img->
        ones_mask = torch.ones_like(pc_voxel).to(pc_voxel.device)
        zeros_mask = torch.zeros_like(pc_voxel).to(pc_voxel.device)
        
        pc_mask = torch.where(pc_voxel!=0, ones_mask, zeros_mask)
        pc_mask, _ = torch.max(pc_mask, dim=1)
        pc_mask = pc_mask.unsqueeze(1)
        img_mask = torch.where(img_voxel!=0, ones_mask, zeros_mask)
        img_mask, _ = torch.max(img_mask, dim=1)
        img_mask = img_mask.unsqueeze(1)

        fused_voxel = pc_mask*img_mask*self.multifuse(torch.cat([self.act(self.multigate(pc_voxel))*img_voxel, pc_voxel], dim=1))
        fused_voxel = fused_voxel + pc_voxel*pc_mask*(1-img_mask) + img_voxel*self.img_fusion(img_voxel, pc_voxel)*(1-pc_mask)*img_mask
        fused_voxel = fused_voxel.view(B,C*Z, Y, X)

        adapt_map = self.adapt_cls(self.adapt_conv(fused_voxel))
        thres_map = pc_mask*img_mask*0.3 + pc_mask*(1-img_mask)*0.5 + (1-pc_mask)*img_mask*0.7 + (1-pc_mask)*(1-img_mask)*1 # size = [B, 1, Z, Y, X]
        thres_map, _ = torch.min(thres_map, dim=2)  # collapse Z-axis, dim=4 size = [B, 1, Y, X]
        
        thres_map = adapt_map * 0.5 + thres_map * 0.5
        pc_dict['spatial_features'] = fused_voxel
        return pc_dict, thres_map


class PointPillarM2FuseV2(nn.Module):
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
        super(PointPillarM2FuseV2, self).__init__()
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
        self.camencode = ImgCamEncode(img_args['img_size'], self.D, self.bevC, self.nx, img_args['chain_channels'], self.downsample, self.grid_conf['ddiscr'], self.grid_conf['mode'], img_args['max_cav'], img_args['grid_att'], img_args['use_depth_gt'], img_args['depth_supervision'])
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
        self.fusion = MultiModalFusion(img_args['bev_dim'])
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
        #self.fusion_net = TemporalFuse(args['collaborative_fusion'])
        #self.fusion_net = TemporalAwareFuse(args['collaborative_fusion'])
        self.fusion_net = AttenComm(args['collaborative_fusion'])
        print("Number of fusion_net parameter: %d" % (sum([param.nelement() for param in self.fusion_net.parameters()])))
        self.fusion_cls_head = []
        # if self.multi_scale:
        #     for num_filters in args['collaborative_fusion']['num_filters']:
        #         self.fusion_cls_head.append(nn.Conv2d(num_filters, args['anchor_number'], kernel_size=1))
        # else:
        #     self.fusion_cls_head = nn.Conv2d(self.outC, args['anchor_number'], kernel_size=1)

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
        # get two types data
        image_inputs_dict = data_dict['image_inputs']
        pc_inputs_dict = data_dict['processed_lidar']
        record_len = data_dict['record_len']

        # process point cloud
        #（1）PillarVFE              pcdet/models/backbones_3d/vfe/pillar_vfe.py
        #（2）PointPillarScatter     pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py
        #（3）BaseBEVBackbone        pcdet/models/backbones_2d/base_bev_backbone.py
        #（4）AnchorHeadSingle       pcdet/models/dense_heads/anchor_head_single.py
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
        B, N, C, imH, imW = x.shape     # torch.Size([4, 1, 3, 320, 480])
        _, x = self.camencode(x, batch_dict['spatial_features_3d'], record_len)     # x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22) -> 多了一个维度D：代表深度
        x = x.view(B, N, self.bevC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)
        # 将图像转换到voxel
        x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 240 x 240
        # 转换到BEV下, collapse Z
        # x = torch.cat(x.unbind(dim=2), 1)  # 消除掉z维
        
        # voxel下的模态融合 img: B*C*Z*Y*X; pc: B*C*Z*Y*X
        batch_dict, thres_map = self.fusion(x, batch_dict)
        batch_dict = self.backbone(batch_dict)
        #backbone = self.backbone if self.modal_multi_scale else None
        #shrink = self.shrink_conv if self.modal_multi_scale else None
        #batch_dict = self.bevbackbone(x, batch_dict, self.num_levels, backbone, shrink)
        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:    # downsample feature to reduce memory
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:    # compressor
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # collaborative fusion
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        # x_fuse, communication_rates, flow_matrix = self.fusion_net(data_dict['lidar_timestamp'], spatial_features_2d, record_len, pairwise_t_matrix, self.bevbackbone, self.cls_head)
        
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                                            batch_dict['spatial_features'],
                                            self.cls_head(spatial_features_2d),
                                            thres_map,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone,
                                            #self.fusion_cls_head)
                                            [self.shrink_conv, self.cls_head, self.reg_head])
            #print('fused_feature: ', fused_feature.shape, communication_rates)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                                            spatial_features_2d,
                                            self.cls_head(spatial_features_2d),
                                            thres_map,
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

    def image_encoding(self, images, features, intrins, extrins):
        B, S, C, H, W = images.shape
        # features = self.camencode(images, )  # torch.Size([B*S, 128, 28, 50])
        B, S, C, Hf, Wf = features.shape
        features = features.view(-1, C, Hf, Wf)
        intrins = intrins.view(-1, 3, 3)
        extrins = extrins.view(-1, 4, 4)

        sy = Hf/float(H)
        sx = Wf/float(W)
        X, Y, Z = self.nx
        bounds = (-int(self.nx[0]/2), int(self.nx[0]/2),
                  -int(self.nx[1]/2), int(self.nx[1]/2),
                  -int(self.nx[2]/2), int(self.nx[2]/2))

        # unproject image feature to 3d grid

        # process intrinsic matrix first
        feat_intrins = camera_utils.scale_intrinsics(intrins, sx, sy)  # torch.Size([B*S, 4, 4])

        # the scene centroid is defined wrt a reference camera, which is usually random
        grid_voxel = cam_unproj.gridcloud3d(1, Z, Y, X, norm=False)
        xyz_camA = cam_unproj.Mem2Ref(grid_voxel, bounds, self.dx, Z, Y, X, assert_cube=False)
        xyz_camA = xyz_camA.to(features.device).repeat(B*S,1,1)
        # xyz_camA = torch.Size([B*S, 320000, 3])

        feat_voxel = cam_unproj.unproject_image_to_mem(features, torch.matmul(feat_intrins, extrins), extrins, Z, Y, X, xyz_camA=xyz_camA)          
        print(feat_voxel.shape)
        # torch.Size([B*S, 128, 200, 8, 200])
        feat_voxel = feat_voxel.view(B, S, C, Z, Y, X) # -> B, S, C, Z, Y, X
        
        # reduce_masked_mean:
        #       feat_voxel and mask_voxel are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
        #       returns shape-1 axis can be a list of axes
        mask_voxel = (torch.abs(feat_voxel) > 0).float()
        assert feat_voxel.shape == mask_voxel.shape

        # while dim = None
        # numer = torch.sum(feat_voxel * mask_voxel)
        # denom = torch.sum(mask_voxel) + 1e-6
        
        # dim=1
        numer = torch.sum(feat_voxel * mask_voxel, dim=1, keepdim=False)
        denom = torch.sum(mask_voxel, dim=1, keepdim=False) + 1e-6

        return numer/denom  # B, C, Z, Y, X. 在S维度加和求平均了，个人理解为多张camera的feature融合了

    #def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
    def get_geometry(self, image_inputs_dict):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
                # process image to get bev
        rots, trans, extrins, intrins, post_rots, post_trans = image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['extrins'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']

        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目) DAIR数据集只有1个相机

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        inv_post_rots = torch.inverse(post_rots.to('cpu')).to(post_rots.device)
        points = inv_post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        #print(intrins.shape)
        inv_intrins = torch.inverse(intrins.to('cpu')).to(intrins.device)
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
