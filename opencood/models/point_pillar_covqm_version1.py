""" 
Author: Co-VQM
"""
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.vqm_modules.pyramid_fuse import PyramidFusion
from opencood.models.vqm_modules.weight_pyramid_fuse import PyramidFusion as SPyramidFusion

from opencood.models.vqm_modules.encodings import build_encoder, ModalFusionBlock
from opencood.models.vqm_modules.proj import pool_feat, match_loss


class PointPillarCoVQM(nn.Module):    
    def __init__(self, args):
        super(PointPillarCoVQM, self).__init__()
        # cuda选择
        self.device = args['device']
        self.max_cav = args['max_cav']
        self.cav_range = args['lidar_range']
        self.unified_score = args['unified_score'] if 'unified_score' in args else False
        self.contrast = args['contrast']

        self.agent_types = args['agents']
        if len(self.agent_types) == 1 and 'a1' in self.agent_types:
            print('Training for homo/hetero-base model.')
        else:
            assert self.max_cav == len(self.agent_types), \
                'max_cav need to be euqal to agent numbers in model.args ' \
                '(We simplify the number of each agent type in the collaborative system to 1)'
        

        self.modality_num_list = []
        for agent_uid in self.agent_types:
            agent_params = 0
            modality_num = 0
            for modality_uid in self.agent_types[agent_uid]:
                if modality_uid in ['origin_hypes', 'model_path', 'fusion_channel', 'origin_fusion_uid', 'freeze_fusion']: continue
                setattr(self, f"{agent_uid}_{modality_uid}", build_encoder(self.agent_types[agent_uid][modality_uid], self.device))
                if 'freeze' in self.agent_types[agent_uid][modality_uid]:
                    setattr(self, f"freeze_{agent_uid}_{modality_uid}", self.agent_types[agent_uid][modality_uid]['freeze'])
                else:
                    setattr(self, f"freeze_{agent_uid}_{modality_uid}", False)

                agent_params += sum([param.nelement() for param in eval(f"self.{agent_uid}_{modality_uid}").parameters()])
                modality_num += 1

            if modality_num > 1:
                assert 'fusion_channel' in self.agent_types[agent_uid]
                setattr(self, f"{agent_uid}_fusion", ModalFusionBlock(self.agent_types[agent_uid]['fusion_channel']))
                if 'freeze_fusion' in self.agent_types[agent_uid]:
                    setattr(self, f"freeze_{agent_uid}_fusion", self.agent_types[agent_uid]['freeze_fusion'])
                else:
                    setattr(self, f"freeze_{agent_uid}_fusion", False)

                agent_params += sum([param.nelement() for param in eval(f"self.{agent_uid}_fusion").parameters()])
            self.modality_num_list.append(modality_num)
            print(f"Number of parameter {agent_uid}: %d" % (agent_params))

        if 'max_modality_agent_index' in args:
            self.max_modality_agent_index = args['max_modality_agent_index'] - 1
        else:
            self.max_modality_agent_index = np.argmax(self.modality_num_list)


        self.fp = args['fp'] if 'fp' in args else True
        self.align_first = args['align_first'] if 'align_first' in args else False
        if len(self.agent_types) <= 1:
            self.fp = False
        if not self.fp:
            print('No FP is used.')
            self.align_block = None
        else:
            print(f'FP is used, and align first is {self.align_first}')
            if self.align_first:
                for idx, agent_uid in enumerate(self.agent_types):
                    if idx != self.max_modality_agent_index:
                        setattr(
                            self,
                            f"{agent_uid}_proj",
                            nn.Embedding(1, args['fusion']['num_filters'][0])
                        )
                self.align_block = None
            else:
                # add single supervision head
                self.align_block = AlignBlock(
                    model_cfg=args['fusion'],
                    agent_types=self.agent_types, 
                    max_modality_agent_index=self.max_modality_agent_index
                )
        # print(self.align_block)

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = args['shrink_header']['use']
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            print("Number of parameter shrink_conv: %d" % (sum([param.nelement() for param in self.shrink_conv.parameters()])))
        

        # if self.unified_score:
        #     self.pyramid_backbone = SPyramidFusion(args['fusion'], self.max_modality_agent_index, self.agent_types, self.fp, not self.align_first)
        # else:
        #     self.pyramid_backbone = PyramidFusion(args['fusion'], self.max_modality_agent_index, self.agent_types, self.fp, not self.align_first)
        if self.unified_score:
            self.pyramid_backbone = SPyramidFusion(args['fusion'])
        else:
            self.pyramid_backbone = PyramidFusion(args['fusion'])
        print("Number of parameter pyramid_backbone: %d" % (sum([param.nelement() for param in self.pyramid_backbone.parameters()])))


        self.cls_head = nn.Conv2d(args['outC'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['outC'], 7 * args['anchor_number'], kernel_size=1)
        self.dir_head = nn.Conv2d(args['outC'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) # BIN_NUM = 2
        if 'seg_upsample' in args:
            self.dynamic_head = nn.ConvTranspose2d(args['outC'], args['seg_class'], kernel_size=2, stride=2, padding=0)
        else:
            self.dynamic_head = nn.Conv2d(args['outC'], args['seg_class'], kernel_size=3, padding=1)

    def freeze_backbone(self, net):
        for p in net.parameters():
            p.requires_grad = False

    def update_model(self, shrink_conv, backbone, cls_head, reg_head, dir_head, dynamic_head, agent_encoders={}, freeze_backbone=False):
        
        self.shrink_conv = shrink_conv
        self.pyramid_backbone = backbone
        self.cls_head = cls_head
        self.reg_head = reg_head
        self.dir_head = dir_head
        freeze_list = [self.shrink_conv, self.pyramid_backbone, self.cls_head, self.reg_head, self.dir_head]
        
        if dynamic_head is not None:
            self.dynamic_head = dynamic_head
            freeze_list.append(self.dynamic_head)
        
        if freeze_backbone:
            for net in freeze_list:
                self.freeze_backbone(net)
            print('backbone network freezed.')
        
        for agent_idx, agent_uid in enumerate(self.agent_types):
            for modality_uid in self.agent_types[agent_uid]:
                if modality_uid in ['origin_hypes', 'model_path', 'fusion_channel', 'origin_fusion_uid', 'freeze_fusion']: continue
                if eval(f"self.freeze_{agent_uid}_{modality_uid}"):
                    assert agent_uid in agent_encoders
                    encoder = agent_encoders[agent_uid]
                    assert f"{agent_uid}_{modality_uid}" in encoder.keys()
                    setattr(self, f"{agent_uid}_{modality_uid}", encoder[f"{agent_uid}_{modality_uid}"])
                    self.freeze_backbone(eval(f"self.{agent_uid}_{modality_uid}"))
                    print(f'{agent_uid}_{modality_uid} net freezed.')

            if self.modality_num_list[agent_idx] > 1:
                if eval(f"self.freeze_{agent_uid}_fusion"):
                    print(encoder.keys())
                    assert 'fusion' in encoder.keys()
                    setattr(self, f"{agent_uid}_fusion", encoder['fusion'])
                    self.freeze_backbone(eval(f"self.{agent_uid}_fusion"))
                    print(f'{agent_uid}_fusion net freezed.')

    def regroup(self, x, record_len, select_idx, unsqueeze=False):
        if self.fp and self.align_first:
            if select_idx != self.max_modality_agent_index:
                x = eval(f"self.a{select_idx+1}_proj").weight.unsqueeze(-1).unsqueeze(-1) * x
        
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

    def minmax_norm(self, data):
        return (data - data.min()) / (data.max() - data.min() + 1e-8) * 2 - 1

    def forward(self, data_dict, mode=[0,1], training=False):
        output_dict = {'pyramid': 'collab'}
        record_len = data_dict['record_len']
        rec_loss, svd_loss, bfp_loss = torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device)
        
        agent_len = 0
        features = []
        ego_features = []
        # process raw data to get feature for each agent
        for agent_uid in self.agent_types:
            f = []
            for modality_uid in self.agent_types[agent_uid]:
                if modality_uid in ['origin_hypes', 'model_path', 'fusion_channel', 'origin_fusion_uid', 'freeze_fusion']: continue
                f.append(
                    self.minmax_norm(
                        eval(f"self.{agent_uid}_{modality_uid}")(data_dict)
                ))

            if len(f) > 1:
                f, _rec_loss, _svd_loss = eval(f"self.{agent_uid}_fusion")(f, mode=mode, training=training)
                rec_loss = rec_loss + _rec_loss
                svd_loss = svd_loss + _svd_loss
            elif len(f) == 1:
                f = f[0]
            else:
                raise

            if len(self.agent_types) == 1:
                features = f
            else:
                select_x, select_ego = self.regroup(f, record_len, select_idx=agent_len)
                if features:
                    for i in range(len(record_len)):
                        if select_x[i] != []:
                            features[i] = torch.cat([features[i], select_x[i]], dim=0)
                else:
                    features = select_x
                if ego_features:
                    for i in range(len(record_len)):
                        ego_features[i] = torch.cat([ego_features[i], select_ego[i]], dim=0)
                else:
                    ego_features = select_ego
            
            agent_len += 1

        if isinstance(features, list):
            features = torch.cat(features, dim=0)


        # # 检查是否包含 NaN, Inf
        # print("Tensor contains NaN:", torch.isnan(features).any().item())
        # print("Tensor contains Inf:", torch.isinf(features).any().item())

        # # back forward projection loss
        # if self.fp and training:
        #     for idx in range(len(record_len)):
        #         agent_num = record_len[idx]
        #         ego_feats = ego_features[idx]

        #         if self.align_first:
        #             for ego_ith_agent in range(agent_num):
        #                 bfp_loss = bfp_loss + match_loss(
        #                     pool_feat(
        #                         f1=ego_feats[ego_ith_agent].unsqueeze(0),
        #                         f2=ego_feats[(ego_ith_agent+1) % ego_feats.shape[0]].unsqueeze(0), 
        #                         pool_dim='hw',
        #                         normalize_feat=True
        #                     ), 
        #                     loss_type='mfro'
        #                 )
                
        #             ego_feats_list = self.pyramid_backbone.get_multiscale_feature(ego_feats)
        #             for level in range(self.pyramid_backbone.num_levels):
        #                 level_features = ego_feats_list[level]
        #                 for ego_ith_agent in range(agent_num):
        #                     f1 = level_features[ego_ith_agent:ego_ith_agent+1]
        #                     if ego_ith_agent != self.max_modality_agent_index:
        #                         f1 = eval(f"self.align_block.a{ego_ith_agent+1}_backward_proj{level}").weight.unsqueeze(-1).unsqueeze(-1) * f1
                            
        #                     if ego_ith_agent + 1 == agent_num:
        #                         f2 = level_features[0:1]
        #                         if self.max_modality_agent_index != 0:
        #                             f2 = eval(f"self.align_block.a1_backward_proj{level}").weight.unsqueeze(-1).unsqueeze(-1) * f2
        #                     else:
        #                         f2 = level_features[ego_ith_agent+1:ego_ith_agent+2]
        #                         if ego_ith_agent + 1 != self.max_modality_agent_index:
        #                             f2 = eval(f"self.align_block.a{ego_ith_agent+2}_backward_proj{level}").weight.unsqueeze(-1).unsqueeze(-1) * f2
                            
        #                     bfp_loss = bfp_loss + match_loss(
        #                         pool_feat(
        #                             f1=f1,
        #                             f2=f2,
        #                             pool_dim='hw',
        #                             normalize_feat=True
        #                         ),
        #                         loss_type='mfro' # abs/mse/mfro/cos
        #                     )
                            
        if self.fp and training:
            for idx in range(len(record_len)):
                agent_num = record_len[idx]
                ego_feats = ego_features[idx]

                if self.align_first:
                    for ego_ith_agent in range(agent_num):
                        bfp_loss = bfp_loss + compute_hetero_loss(
                            f1=ego_feats[ego_ith_agent].unsqueeze(0),
                            f2=ego_feats[(ego_ith_agent+1) % ego_feats.shape[0]].unsqueeze(0),
                            loss_type='mfro'
                        )
                
                    ego_feats_list = self.pyramid_backbone.get_multiscale_feature(ego_feats)
                    for level in range(self.pyramid_backbone.num_levels):
                        level_features = ego_feats_list[level]
                        for ego_ith_agent in range(agent_num):
                            f1 = level_features[ego_ith_agent:ego_ith_agent+1]
                            if ego_ith_agent != self.max_modality_agent_index:
                                f1 = eval(f"self.align_block.a{ego_ith_agent+1}_backward_proj{level}").weight.unsqueeze(-1).unsqueeze(-1) * f1
                            
                            if ego_ith_agent + 1 == agent_num:
                                f2 = level_features[0:1]
                                if self.max_modality_agent_index != 0:
                                    f2 = eval(f"self.align_block.a1_backward_proj{level}").weight.unsqueeze(-1).unsqueeze(-1) * f2
                            else:
                                f2 = level_features[ego_ith_agent+1:ego_ith_agent+2]
                                if ego_ith_agent + 1 != self.max_modality_agent_index:
                                    f2 = eval(f"self.align_block.a{ego_ith_agent+2}_backward_proj{level}").weight.unsqueeze(-1).unsqueeze(-1) * f2
                            
                            bfp_loss = bfp_loss + match_loss(
                                pool_feat(
                                    f1=f1,
                                    f2=f2,
                                    pool_dim='hw',
                                    normalize_feat=True
                                ),
                                loss_type='mfro' # abs/mse/mfro/cos
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
                                                )
        else:
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                    features,
                                                    record_len, 
                                                    affine_matrix,
                                                    self.align_block, 
                                                    self.fp, 
                                                    not self.align_first, 
                                                    self.max_modality_agent_index
                                                )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        # dir_preds = self.dir_head(fused_feature)
        seg_preds = self.dynamic_head(fused_feature)

        # output
        output_dict.update({
            'rec_loss': rec_loss,
            'svd_loss': svd_loss,
            'bfp_loss': bfp_loss,
        })

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            # 'dir_preds': dir_preds,
            'seg_preds': seg_preds,
            'psm': cls_preds,
            'rm': reg_preds,
        })

        if self.max_modality_agent_index != 0:
        # if True:
            single_dict = self.inference_single(data_dict, output_agent_index=0)
        
            output_dict.update({
                # 'cls_preds': cls_preds,
                # 'reg_preds': reg_preds,
                # 'dir_preds': dir_preds,
                'seg_preds_single': single_dict['seg_preds'],
                'psm_single': single_dict['psm'],
                'rm_single': single_dict['rm'],
            })
        return output_dict

    def inference_single(self, data_dict, output_agent_index):
        output_dict = {'pyramid': 'collab'}
        record_len = data_dict['record_len']
        # rec_loss, svd_loss, bfp_loss = torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device)
        
        features = []
        # process raw data to get feature for each agent
        for agent_idx, agent_uid in enumerate(self.agent_types):
            if output_agent_index != agent_idx: continue

            f = []
            for modality_uid in self.agent_types[agent_uid]:
                if modality_uid in ['origin_hypes', 'model_path', 'fusion_channel', 'origin_fusion_uid', 'freeze_fusion']: continue
                f.append(
                    self.minmax_norm(
                        eval(f"self.{agent_uid}_{modality_uid}")(data_dict)
                ))

            if len(f) > 1:
                f, _rec_loss, _svd_loss = eval(f"self.{agent_uid}_fusion")(f, mode=[0,1], training=False)
                # rec_loss = rec_loss + _rec_loss
                # svd_loss = svd_loss + _svd_loss
            elif len(f) == 1:
                f = f[0]
            else:
                raise
            
            # select the ego as single feature
            if len(record_len) > 1:
                features = torch.cat(self.regroup(f, record_len, 0)[1], dim=0)
            else:
                features = f[0:1]

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        align_block = None if output_agent_index == self.max_modality_agent_index else self.align_block
        fused_feature, occ_outputs = self.pyramid_backbone.forward_single(features, output_agent_index, align_block)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        # dir_preds = self.dir_head(fused_feature)
        seg_preds = self.dynamic_head(fused_feature)

        # output
        # output_dict.update({
        #     'rec_loss': rec_loss,
        #     'svd_loss': svd_loss,
        #     'bfp_loss': bfp_loss,
        # })

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            # 'dir_preds': dir_preds,
            'seg_preds': seg_preds,
            'psm': cls_preds,
            'rm': reg_preds,
        })

        return output_dict


class AlignBlock(nn.Module):
    def __init__(self, model_cfg, agent_types, max_modality_agent_index):
        super(AlignBlock, self).__init__()
        # add single supervision head
        for i in range(len(model_cfg['layer_nums'])):
            for agent_idx, agent_uid in enumerate(agent_types):
                if agent_idx != max_modality_agent_index:
                    setattr(
                        self,
                        f"{agent_uid}_backward_proj{i}",
                        nn.Embedding(1, model_cfg["num_filters"][i]),
                    )
    
    # def forward(self, x):
    #     return x


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