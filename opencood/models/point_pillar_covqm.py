""" 
Author: Co-VQM
"""

import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.vqm_modules.pyramid_fuse import PyramidFusion
from opencood.models.vqm_modules.weight_pyramid_fuse import PyramidFusion as SPyramidFusion

from opencood.models.vqm_modules.f1 import CoVQMF1
from opencood.models.vqm_modules.f2 import CoVQMF2
from opencood.models.vqm_modules.f3 import CoVQMF3
from opencood.models.vqm_modules.proj import pool_feat, match_loss


class PointPillarCoVQM(nn.Module):    
    def __init__(self, args, f1_net=None, f2_net=None, f3_net=None):
        super(PointPillarCoVQM, self).__init__()
        # cuda选择
        self.device = args['device']
        self.max_cav = args['max_cav']
        self.cav_range = args['lidar_range']
        self.supervise_single = args['supervise_single'] if 'supervise_single' in args else False
        self.unified_score = args['unified_score'] if 'unified_score' in args else False
        self.bfp = args['bfp'] if 'bfp' in args else True
        if not self.bfp:
            print('No BFP is used.')

        self.agent_len = 0
        self.f1 = None
        if f1_net is not None:
            self.f1 = f1_net
        if 'f1' in args:
            if self.f1 is not None:
                if 'freeze' in args['f1'] and args['f1']['freeze']:
                    self.freeze_backbone(self.f1) 
                    print('f1 net freezed.')
            else:
                self.f1 = CoVQMF1(args['f1'])
            self.agent_len += 1
            print("Number of parameter f1: %d" % (sum([param.nelement() for param in self.f1.parameters()])))
        if self.agent_len > self.max_cav:
            self.agent_len -= 1
            self.f1 = None
        

        self.f2 = None
        if f2_net is not None:
            self.f2 = f2_net
        if 'f2' in args:
            if self.f2 is not None:
                if 'freeze' in args['f2'] and args['f2']['freeze']:
                    self.freeze_backbone(self.f2)
                    print('f2 net freezed.')
            else:
                self.f2 = CoVQMF2(args['f2'])
            self.agent_len += 1
            print("Number of parameter f2: %d" % (sum([param.nelement() for param in self.f2.parameters()])))
        if self.agent_len > self.max_cav:
            self.agent_len -= 1
            self.f2 = None
        self.f2_proj = nn.Conv2d(args['fusion']['num_filters'][0], args['fusion']['num_filters'][0], kernel_size=3, stride=1, padding=1)

        self.f3 = None
        if f3_net is not None:
            self.f3 = f3_net
        if 'f3' in args:
            if self.f3 is not None:
                if 'freeze' in args['f3'] and args['f3']['freeze']:
                    self.freeze_backbone(self.f3)
                    print('f3 net freezed.')
            else:
                self.f3 = CoVQMF3(args['f3'])
            self.agent_len += 1
            print("Number of parameter f3: %d" % (sum([param.nelement() for param in self.f3.parameters()])))
        if self.agent_len > self.max_cav:
            self.agent_len -= 1
            self.f3 = None
        self.f3_proj = nn.Conv2d(args['fusion']['num_filters'][0], args['fusion']['num_filters'][0], kernel_size=3, stride=1, padding=1)


        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = args['shrink_header']['use']
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            print("Number of parameter shrink_conv: %d" % (sum([param.nelement() for param in self.shrink_conv.parameters()])))
        
        if self.unified_score:
            self.pyramid_backbone = SPyramidFusion(args['fusion'])
        else:
            self.pyramid_backbone = PyramidFusion(args['fusion'])
        print("Number of parameter pyramid_backbone: %d" % (sum([param.nelement() for param in self.pyramid_backbone.parameters()])))

        self.cls_head = nn.Conv2d(args['outC'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['outC'], 7 * args['anchor_number'], kernel_size=1)
        self.dir_head = nn.Conv2d(args['outC'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) # BIN_NUM = 2

    def freeze_backbone(self, net):
        for p in net.parameters():
            p.requires_grad = False

    def regroup(self, x, record_len, select_idx, unsqueeze=False, bfp=True):
        if bfp:
            if select_idx == 1:
                x = self.f2_proj(x)
            elif select_idx == 2:
                x = self.f3_proj(x)
        if unsqueeze:
            x = x.unsqueeze(1)
        split_x = torch.tensor_split(x, torch.cumsum(record_len, dim=0)[:-1].cpu())

        select_x, select_ego = [], []
        for _x in split_x:
            if _x.shape[0] < select_idx + 1:
                select_x.append([])
            else:
                select_x.append(_x[select_idx:select_idx+1])
            select_ego.append(_x[0:1])
        return select_x, select_ego

    def forward(self, data_dict, mode=[0,1], training=True):
        output_dict = {'pyramid': 'collab'}
        record_len = data_dict['record_len']
        rec_loss, svd_loss, bfp_loss = torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device)
        m_len = 1

        features = []
        ego_features = []
        modality_len = []
        # process agent 1 data to get feature f1
        # f1: (b,m,c,h,w)
        if self.f1 is not None:
            batch_dict = {
                'voxel_features': data_dict['processed_lidar']['voxel_features'],
                'voxel_coords': data_dict['processed_lidar']['voxel_coords'],
                'voxel_num_points': data_dict['processed_lidar']['voxel_num_points'],
                'image_inputs': data_dict['image_inputs'],
                'batch_size': torch.sum(record_len).cpu().numpy(),
                'record_len': record_len
            }
            f, m_len, rec_loss, svd_loss = self.f1(batch_dict, mode=mode, training=training)
            
            features, ego_features = self.regroup(f, record_len, select_idx=len(modality_len))
            if self.agent_len <= 1:
                features = f
            modality_len.append(m_len)
        
        # process agent 2 data to get feature f2
        # (b,c,h,w)
        if self.f2 is not None:
            batch_dict = {
                'voxel_features': data_dict['processed_lidar2']['voxel_features'],
                'voxel_coords': data_dict['processed_lidar2']['voxel_coords'],
                'voxel_num_points': data_dict['processed_lidar2']['voxel_num_points'],
                'batch_size': torch.sum(record_len).cpu().numpy(),
                'record_len': record_len
            }
            f = self.f2(batch_dict)
            
            if self.agent_len <= 1:
                features = f.unsqueeze(1)
            else:
                select_x, select_ego = self.regroup(f, record_len, select_idx=len(modality_len), unsqueeze=True, bfp=self.bfp)
                if features:
                    for i in range(len(record_len)):
                        if select_x[i] != []:
                            features[i] = torch.cat([features[i], select_x[i].repeat(1,m_len,1,1,1)], dim=0)
                else:
                    features = select_x
            
                if ego_features:
                    for i in range(len(record_len)):
                        ego_features[i] = torch.cat([ego_features[i], select_ego[i]], dim=1)
                else:
                    ego_features = select_ego
            
            modality_len.append(1)

        # process agent 3 data to get feature f3
        # (b,c,h,w)
        if self.f3 is not None:
            f = self.f3(data_dict['image_inputs'])
            
            if self.agent_len <= 1:
                features = f.unsqueeze(1)
            else:
                select_x, select_ego = self.regroup(f, record_len, select_idx=len(modality_len), unsqueeze=True, bfp=self.bfp)
                if features:
                    for i in range(len(record_len)):
                        if select_x[i] != []:
                            features[i] = torch.cat([features[i], select_x[i].repeat(1,m_len,1,1,1)], dim=0)
                else:
                    features = select_x
            
                if ego_features:
                    for i in range(len(record_len)):
                        ego_features[i] = torch.cat([ego_features[i], select_ego[i]], dim=1)
                else:
                    ego_features = select_ego
            
            modality_len.append(1)

        if self.agent_len > 1:
            features = torch.cat(features, dim=0)

        # back forward projection loss
        # if self.bfp and training and (self.agent_len > 1 or (self.f1 is not None and m_len > 1)):
        if self.bfp and training and self.agent_len > 1:
            for idx in range(len(record_len)):
                ego_feats = ego_features[idx]
                for ego_idx in range(ego_feats.shape[1]):
                    bfp_loss = bfp_loss + match_loss(
                        pool_feat(
                            f1=ego_feats[:, ego_idx], 
                            f2=ego_feats[:, (ego_idx+1) % ego_feats.shape[1]], 
                            pool_dim='hw',
                            normalize_feat=True
                        ), 
                        loss_type='mfro'
                    )

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        if self.unified_score:
            fused_feature_single = self.pyramid_backbone.resnet._forward_impl(
                rearrange(features, 'b m c h w -> (b m) c h w'), 
                return_interm=False
            )
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                    features,
                                                    self.cls_head(fused_feature_single).sigmoid().max(dim=1)[0].unsqueeze(1),
                                                    record_len, 
                                                    affine_matrix, 
                                                    None, 
                                                    None
                                                )
        else:    
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                    features,
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

        # output
        output_dict.update({
            'rec_loss': rec_loss,
            'svd_loss': svd_loss,
            'bfp_loss': bfp_loss,
        })

        if self.f1 is not None and training and m_len > 1:
            output_dict.update({'modality_num': m_len})

            # (b*m_len,c,h,w) -> (b,m_len,c,h,w)
            cls_preds = rearrange(cls_preds, '(b m) c h w -> b m c h w', b=len(record_len), m=m_len+1)
            reg_preds = rearrange(reg_preds, '(b m) c h w -> b m c h w', b=len(record_len), m=m_len+1)
            dir_preds = rearrange(dir_preds, '(b m) c h w -> b m c h w', b=len(record_len), m=m_len+1)
            
            for x_idx in range(m_len):
                output_dict.update({
                    'cls_preds_{}'.format(x_idx): cls_preds[:,x_idx],
                    'reg_preds_{}'.format(x_idx): reg_preds[:,x_idx],
                    'dir_preds_{}'.format(x_idx): dir_preds[:,x_idx],
                    # 'occ_single_list_{}'.format(x_idx): eval(f"split_occ_outputs{x_idx}"),
                })

            output_dict.update({
                'cls_preds': cls_preds[:,-1],
                'reg_preds': reg_preds[:,-1],
                'dir_preds': dir_preds[:,-1],
                # 'psm': cls_preds[:,-1],
                # 'rm': reg_preds[:,-1],
            })

            return output_dict

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            # 'psm': cls_preds,
            # 'rm': reg_preds,
        })

        return output_dict


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