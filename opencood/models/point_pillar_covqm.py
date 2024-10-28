""" 
Author: Co-VQM
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.heal_modules.pyramid_fuse import PyramidFusion
from opencood.models.vqm_modules.weight_pyramid_fuse import PyramidFusion as SPyramidFusion

from opencood.models.vqm_modules.encodings import build_encoder, ModalFusionBlock
from opencood.models.heal_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.vqm_modules.proj import cal_bfp_loss


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# visualization = False
tsne_points = []
tsne_labels = []
tsne_colors = []


def plot_embedding_2D(data, label, colors, title):
    print(type(data))
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(colors[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    plt.savefig('tsne.png')
    quit()


class PointPillarCoVQM(nn.Module):    
    def __init__(self, args):
        super(PointPillarCoVQM, self).__init__()
        # cuda选择
        self.device = args['device']
        self.max_cav = args['max_cav']
        self.cav_range = args['lidar_range']
        self.unified_score = args['unified_score'] if 'unified_score' in args else False
        self.freeze_backbone = args['freeze_backbone'] if 'freeze_backbone' in args else False
        self.no_collab = args['no_collab'] if 'no_collab' in args else False
        self.supervise_ego = args['supervise_ego'] if 'supervise_ego' in args else False
        self.back_proj = args['back_proj'] if 'back_proj' in args else False
        print('no_collab, supervise_ego, back_proj: ', self.no_collab, self.supervise_ego, self.back_proj)

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
                if modality_uid in ['origin_hypes', 'model_path', 'fusion_mode', 'fusion_channel', 'origin_fusion_uid', 'freeze_fusion']: continue
                setattr(self, f"{agent_uid}_{modality_uid}", build_encoder(self.agent_types[agent_uid][modality_uid], self.device))
                if 'freeze' in self.agent_types[agent_uid][modality_uid]:
                    setattr(self, f"freeze_{agent_uid}_{modality_uid}", self.agent_types[agent_uid][modality_uid]['freeze'])
                else:
                    setattr(self, f"freeze_{agent_uid}_{modality_uid}", False)

                agent_params += sum([param.nelement() for param in eval(f"self.{agent_uid}_{modality_uid}").parameters()])
                modality_num += 1

            if modality_num > 1:
                assert 'fusion_channel' in self.agent_types[agent_uid]
                setattr(self, f"{agent_uid}_fusion", ModalFusionBlock(self.agent_types[agent_uid]['fusion_channel'], self.agent_types[agent_uid]['fusion_mode']))
                if 'freeze_fusion' in self.agent_types[agent_uid]:
                    setattr(self, f"freeze_{agent_uid}_fusion", self.agent_types[agent_uid]['freeze_fusion'])
                else:
                    setattr(self, f"freeze_{agent_uid}_fusion", False)

                agent_params += sum([param.nelement() for param in eval(f"self.{agent_uid}_fusion").parameters()])
            self.modality_num_list.append(modality_num)
            print(f"Number of parameter {agent_uid}: %d" % (agent_params))


        if len(self.agent_types) > 1:
            assert 'proj_backbone_args' in args
            self.proj_base_agent = args['proj_base_agent'] if 'proj_base_agent' in args else True
            if self.proj_base_agent:
                print('Proj base agent.')
            else:
                # self.freeze_backbone = True
                print('Base agent will not be projected.')
            for idx, agent_uid in enumerate(self.agent_types):
                # if idx == 0: continue
                setattr(self, f"{agent_uid}_proj", ResNetBEVBackbone(args['proj_backbone_args']))
        
            assert 'align_loss' in args
            self.loss_type = args['align_loss']
            if args['align_loss'] == 'abs':
                self.distribution_loss = nn.L1Loss()
            elif args['align_loss'] == 'mse':
                self.distribution_loss = nn.MSELoss()
            elif args['align_loss'] == 'kl':
                self.distribution_loss = nn.KLDivLoss(reduction='batchmean')
                self.loss_type = 'cos'
                self.distribution_loss = nn.CosineSimilarity(dim=0)
            elif args['align_loss'] == 'abs+cos':
                self.distribution_loss = [
                    nn.L1Loss(),
                    nn.CosineSimilarity(dim=0)
                ]
            # else:
            #     raise

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
        if 'seg_upsample' in args:
            self.dynamic_head = nn.ConvTranspose2d(args['outC'], args['seg_class'], kernel_size=2, stride=2, padding=0)
        else:
            self.dynamic_head = None # nn.Conv2d(args['outC'], args['seg_class'], kernel_size=3, padding=1)

        if self.back_proj:
            assert 'bfp_loss' in args
            self.bfp_loss_type = args['bfp_loss']
            self.ego_backbone = []
            self.back_embed = nn.Conv2d(args['outC'], args['outC'], kernel_size=1)
            # self.back_embed = nn.Embedding(1, args['outC'])

    def freeze_networks(self, net):
        for p in net.parameters():
            p.requires_grad = False
    
    def inference_freeze(self):
        # if self.freeze_backbone:
        #     freeze_list = [self.shrink_conv, self.pyramid_backbone, self.cls_head, self.reg_head, self.dir_head]
        #     if self.dynamic_head is not None:
        #         freeze_list.append(self.dynamic_head)
        #     for net in freeze_list:
        #         self.freeze_networks(net)
        #     print('backbone network freezed.')
        
        # for agent_idx, agent_uid in enumerate(self.agent_types):
        #     for modality_uid in self.agent_types[agent_uid]:
        #         if modality_uid in ['origin_hypes', 'model_path', 'fusion_channel', 'origin_fusion_uid', 'freeze_fusion']: continue
        #         if eval(f"self.freeze_{agent_uid}_{modality_uid}"):
        #             self.freeze_networks(eval(f"self.{agent_uid}_{modality_uid}"))
        #             print(f'{agent_uid}_{modality_uid} net freezed.')

        #     if self.modality_num_list[agent_idx] > 1:
        #         if eval(f"self.freeze_{agent_uid}_fusion"):
        #             self.freeze_networks(eval(f"self.{agent_uid}_fusion"))
        #             print(f'{agent_uid}_fusion net freezed.')
        self.freeze_networks(self)

    def update_model(self, shrink_conv, backbone, cls_head, reg_head, dir_head, dynamic_head, agent_encoders={}):
        self.shrink_conv = shrink_conv
        self.pyramid_backbone = backbone
        self.cls_head = cls_head
        self.reg_head = reg_head
        self.dir_head = dir_head
        freeze_list = [self.shrink_conv, self.pyramid_backbone, self.cls_head, self.reg_head, self.dir_head]
        
        if dynamic_head is not None:
            self.dynamic_head = dynamic_head
            freeze_list.append(self.dynamic_head)
        else:
            self.dynamic_head = None
        
        if self.freeze_backbone:
            for net in freeze_list:
                self.freeze_networks(net)
            print('backbone network freezed.')
        
        if self.back_proj:
            self.ego_backbone = [shrink_conv, backbone]
            for net in self.ego_backbone:
                self.freeze_networks(net)
        
        for agent_idx, agent_uid in enumerate(self.agent_types):
            for modality_uid in self.agent_types[agent_uid]:
                if modality_uid in ['origin_hypes', 'model_path', 'fusion_mode', 'fusion_channel', 'origin_fusion_uid', 'freeze_fusion']: continue
                if eval(f"self.freeze_{agent_uid}_{modality_uid}"):
                    assert agent_uid in agent_encoders
                    encoder = agent_encoders[agent_uid]
                    assert f"{agent_uid}_{modality_uid}" in encoder.keys()
                    setattr(self, f"{agent_uid}_{modality_uid}", encoder[f"{agent_uid}_{modality_uid}"])
                    self.freeze_networks(eval(f"self.{agent_uid}_{modality_uid}"))
                    print(f'{agent_uid}_{modality_uid} net freezed.')

            if self.modality_num_list[agent_idx] > 1:
                if eval(f"self.freeze_{agent_uid}_fusion"):
                    print(encoder.keys())
                    assert 'fusion' in encoder.keys()
                    setattr(self, f"{agent_uid}_fusion", encoder['fusion'])
                    self.freeze_networks(eval(f"self.{agent_uid}_fusion"))
                    print(f'{agent_uid}_fusion net freezed.')

    def regroup(self, x, record_len, select_idx):
        # if select_idx != 0:
        #     x = eval(f"self.a{select_idx+1}_proj")({'spatial_features':x})['spatial_features_2d']

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

    def forward(self, data_dict, mode=[0,1], training=False, visualization=False):
        return self.forward_collab(data_dict, mode, training, visualization)
    
    def forward_collab(self, data_dict, mode=[0,1], training=False, visualization=False):
        output_dict = {'pyramid': 'collab'}
        record_len = data_dict['record_len']
        rec_loss, svd_loss, bfp_loss, align_loss = \
            torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device),\
            torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device)

        agent_len = 0
        origin_features = []
        proj_features = []

        features = []
        # ego_features = []
        # process raw data to get feature for each agent
        for agent_idx, agent_uid in enumerate(self.agent_types):
            f = []
            for modality_uid in self.agent_types[agent_uid]:
                if modality_uid in ['origin_hypes', 'model_path', 'fusion_mode', 'fusion_channel', 'origin_fusion_uid', 'freeze_fusion']: continue
                f.append(
                    self.minmax_norm(
                        eval(f"self.{agent_uid}_{modality_uid}")(data_dict)
                ))

            if len(f) > 1:
                f, _rec_loss, _svd_loss = eval(f"self.{agent_uid}_fusion")(f, mode=mode, training=training)
                # rec_loss & svd_loss
                rec_loss = rec_loss + _rec_loss
                svd_loss = svd_loss + _svd_loss
                f = self.minmax_norm(f)
            elif len(f) == 1:
                f = f[0]
            else:
                raise

            # len(origin_features) == len(proj_features) == agent number
            origin_features.append(f)
            if len(self.agent_types) == 1:
                features = f
            else:
                if agent_idx == 0 and not self.proj_base_agent:
                    pass
                else:
                    f = eval(f"self.a{agent_idx+1}_proj")({'spatial_features':f})['spatial_features_2d']
                proj_features.append(f)
                select_x, _ = self.regroup(f, record_len, select_idx=agent_len)
                # select_x, select_ego = self.regroup(f, record_len, select_idx=agent_len)
                if features:
                    for i in range(len(record_len)):
                        if select_x[i] != []:
                            features[i] = torch.cat([features[i], select_x[i]], dim=0)
                else:
                    features = select_x
                # if ego_features:
                #     for i in range(len(record_len)):
                #         ego_features[i] = torch.cat([ego_features[i], select_ego[i]], dim=0)
                # else:
                #     ego_features = select_ego
            
            agent_len += 1

        # align loss
        if len(self.agent_types) > 1 and training:
            for idx in range(len(proj_features)):
                if self.loss_type == 'cos':
                    dist_loss = self.distribution_loss(
                        proj_features[idx],
                        proj_features[(idx+1) % len(proj_features)]
                    )
                    dist_loss = 1 - torch.mean(dist_loss)
                elif self.loss_type == 'kl':
                    dist_loss = 0.5 * self.distribution_loss(
                            F.log_softmax(proj_features[idx], dim=1),
                            F.softmax(proj_features[(idx+1) % len(proj_features)], dim=1)
                        ) + 0.5 * self.distribution_loss(
                            F.log_softmax(proj_features[(idx+1) % len(proj_features)], dim=1),
                            F.softmax(proj_features[idx], dim=1)
                        )
                elif self.loss_type == 'mse' or self.loss_type == 'abs':
                    dist_loss = self.distribution_loss(
                        proj_features[idx],
                        proj_features[(idx+1) % len(proj_features)]
                    )
                elif self.loss_type == 'abs+cos':
                    dist_loss = self.distribution_loss[0](
                        proj_features[idx],
                        proj_features[(idx+1) % len(proj_features)]
                    ) + torch.mean(0.5 - 0.5 * self.distribution_loss[0](
                        proj_features[idx],
                        proj_features[(idx+1) % len(proj_features)]
                    ))
                else:
                    # raise
                    dist_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                align_loss = align_loss + dist_loss

        # visualization of shared-specific fts here
        if visualization:
            # collecting enough data points
            # tsne_2D = TSNE(n_components=2, random_state=0)
            tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
            symbols = ['x', 'o', '+', '*', '=', '/']
            
            vis_ft = []
            vis_ft_clr = []
            
            # if ego_features:
            #     assert len(ego_features[0]) == 2
            #     for idx, ft in enumerate(ego_features[0]):
            #         vis_ft.append(ft)
            #         vis_ft_clr.append(idx)

            #     vis_ft = torch.stack(vis_ft)
            #     vis_ft = vis_ft.view(vis_ft.shape[0], -1)

            #     tsne_points.append(vis_ft.cpu().detach().numpy())
            #     tsne_colors.append(vis_ft_clr)
            #     tsne_labels.append(['x', 'o'])
            # else:
            #     assert features.shape[0] == 2
            #     vis_ft = features.view(features.shape[0], -1)
            #     tsne_points.append(vis_ft.cpu().detach().numpy())
            #     tsne_colors.append([0, 1])
            #     tsne_labels.append(['x', 'o'])

            # only 1 type agent
            if len(self.agent_types) > 1:
                assert len(origin_features) > 1
                assert len(proj_features) > 1                

                for idx, ft in enumerate(origin_features):
                    vis_ft.append(ft[0])
                    vis_ft_clr.append(idx)

                for idx, ft in enumerate(proj_features):
                    vis_ft.append(ft[0])
                    vis_ft_clr.append(idx+len(origin_features))
            
            else:
                assert len(origin_features) == 1
                
                vis_ft.append(origin_features[0])
                vis_ft_clr.append(0)

            vis_ft = torch.stack(vis_ft)
            vis_ft = vis_ft.view(vis_ft.shape[0], -1)
            tsne_points.append(vis_ft.cpu().detach().numpy())
            tsne_colors.append(vis_ft_clr)
            tsne_labels.append(symbols[:len(vis_ft)])

            if len(tsne_points) == 40:  # actual visualization
                vis_tsne_points = np.concatenate(tsne_points)
                vis_tsne_labels = np.concatenate(tsne_labels)
                vis_tsne_colors = np.concatenate(tsne_colors)
                vis_tsne_2D = tsne_2D.fit_transform(vis_tsne_points)
                
                plot_embedding_2D(vis_tsne_2D, vis_tsne_labels, vis_tsne_colors, 't-SNE of Features')
                # plt.show()

        if isinstance(features, list):
            features = torch.cat(features, dim=0)

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
                                                    affine_matrix
                                                )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        # dir_preds = self.dir_head(fused_feature)
        seg_preds = self.dynamic_head(fused_feature)

        # pred for multi-agent (collab loss)
        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            # 'dir_preds': dir_preds,
            'seg_preds': seg_preds,
            'psm': cls_preds,
            'rm': reg_preds,
        })

        # pred for ego agent (ego performance loss)
        if self.supervise_ego and len(self.agent_types) > 1 and training:
            ego_features, _ = self.pyramid_backbone.forward_collab(
                proj_features[0],
                record_len, 
                affine_matrix
            )
            if self.shrink_flag:
                ego_features = self.shrink_conv(ego_features)

            output_dict.update({
                # 'cls_preds_ego': ego_output_dict['cls_preds'],
                # 'reg_preds_ego': ego_output_dict['reg_preds'],
                # 'dir_preds_ego': self.dir_head(fused_feature),
                'seg_preds_ego': self.dynamic_head(ego_features),
                'psm_ego': self.cls_head(ego_features),
                'rm_ego': self.reg_head(ego_features),
            })

            # backward proj loss
            if self.back_proj:
                ego_features = self.back_embed(ego_features)
                # ego_features = ego_features * self.back_embed.weight.unsqueeze(-1).unsqueeze(-1)
                
                assert len(self.ego_backbone) > 0
                # = pyramid_backbone
                origin_ego_features, _ = self.ego_backbone[1].forward_collab(
                    origin_features[0],
                    record_len, 
                    affine_matrix
                )
                if self.shrink_flag:
                    origin_ego_features = self.ego_backbone[0](origin_ego_features)
                
                bfp_loss = cal_bfp_loss(ego_features, origin_ego_features, self.bfp_loss_type)

        # collect loss
        output_dict.update({
            'rec_loss': rec_loss,
            'svd_loss': svd_loss,
            'bfp_loss': bfp_loss,
            'align_loss': align_loss,
        })

        return output_dict

    def inference_single(self, data_dict, output_agent_index):
        output_dict = {'pyramid': 'collab'}
        record_len = data_dict['record_len']
        # rec_loss, svd_loss, bfp_loss = torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device), torch.tensor(0.0, requires_grad=True).to(self.device)
        
        features = []
        # process raw data to get feature for each agent
        for agent_idx, agent_uid in enumerate(self.agent_types):
            if agent_idx != output_agent_index: continue

            f = []
            for modality_uid in self.agent_types[agent_uid]:
                if modality_uid in ['origin_hypes', 'model_path', 'fusion_mode', 'fusion_channel', 'origin_fusion_uid', 'freeze_fusion']: continue
                f.append(
                    self.minmax_norm(
                        eval(f"self.{agent_uid}_{modality_uid}")(data_dict)
                ))

            if len(f) > 1:
                f, _, _ = eval(f"self.{agent_uid}_fusion")(f, mode=[0,1], training=False)
            elif len(f) == 1:
                f = f[0]
            else:
                raise
            # batch_size=1 in inference, select the ego as single feature
            features = f[0:1]
            # if agent length > 1, the scenario is hetero collaboration, proj exists
            if len(self.agent_types) > 1:
                features = eval(f"self.a{agent_idx+1}_proj")({'spatial_features':features})['spatial_features_2d']


        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        # fused_feature, occ_outputs = self.pyramid_backbone.forward_single(features)

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        _record_len = torch.ones(record_len.shape).long().to(record_len.device)
        # _record_len = record_len
        if self.unified_score:
            fused_feature_single = self.pyramid_backbone.resnet._forward_impl(
                rearrange(features, 'b m c h w -> (b m) c h w'), 
                return_interm=False
            )
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                    features,
                                                    self.cls_head(fused_feature_single).sigmoid().max(dim=1)[0].unsqueeze(1),
                                                    _record_len, 
                                                    affine_matrix, 
                                                )
        else:
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                    features,
                                                    _record_len, 
                                                    affine_matrix
                                                )

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