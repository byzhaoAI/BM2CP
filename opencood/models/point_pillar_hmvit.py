""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor

from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization

from opencood.models.hmvit_modules.spatial_transformation import SpatialTransformation
from opencood.models.hmvit_modules.base_camera_lidar_intermediate import BaseCameraLiDARIntermediate
from opencood.models.hmvit_modules.naive_decoder import NaiveDecoder
from opencood.models.hmvit_modules.hetero_decoder import HeteroDecoder
from opencood.models.hmvit_modules.hetero_fusion import HeteroFusionBlock
from opencood.models.hmvit_modules.base_transformer import HeteroFeedForward

from opencood.models.hmvit_modules import types_encoding

from einops import rearrange

import torchvision

from opencood.models.vqm_modules.f1 import CoVQMF1
from opencood.models.vqm_modules.encodings.second import SECOND


class HeteroFusion(nn.Module):
    def __init__(self, config):
        super(HeteroFusion, self).__init__()
        self.spatial_transform = SpatialTransformation(
            config['spatial_transform'])
        self.downsample_rate = config['spatial_transform']['downsample_rate']
        self.discrete_ratio = config['spatial_transform']['voxel_size'][0]



        self.hetero_fusion_block = HeteroFusionBlock(
            config['hetero_fusion_block'])
        input_dim = config['hetero_fusion_block']['input_dim']

        self.num_iters = config['num_iters']
        self.mlp_head = HeteroFeedForward(input_dim, input_dim, 0)

    def forward(self, x, pairwise_t_matrix, mode, record_len, mask):
        temp = mode.detach().clone()

        for _ in range(self.num_iters):
            x = self.hetero_fusion_block(x, pairwise_t_matrix, temp,
                                         record_len, mask)
        # x = x[:, 0, ...]
        # (B, M, C, H, W) -> (B, C, H, W)
        x = x[:, 0, ...].permute(0, 2, 3, 1)
        x = self.mlp_head(x.unsqueeze(1), temp[:, :1]).squeeze(1).permute(0, 3, 1, 2)
        return x


class PointPillarHMViT(nn.Module):
    def __init__(self, args):
        super(PointPillarHMViT, self).__init__()
        self.max_cav = args['max_cav']
        
        self.type = 0
        if 'multimodal' in args:
            self.basebone = types_encoding.Multimodal(args)
            self.type += 1
        else:
            self.basebone = None
        
        if 'pillar' in args:
            self.pillar = types_encoding.PointPillar(args)
            self.type += 1
        else:
            self.pillar = None

        if 'second' in args:
            self.second = types_encoding.SECOND(args)
            self.type += 1
        else:
            self.second = None

        if self.type > 1:
            assert 'feat_proj' in args
            self.feat_proj = args['feat_proj'] if 'feat_proj' in args else False
            print('feature proj: ', self.feat_proj)
            if self.feat_proj:
                assert 'proj_backbone_args' in args
                for idx in range(self.type):
                    setattr(self, f"a{idx}_proj", HEALResNetBEVBackbone(args['proj_backbone_args']))

        # self.compression = args['compression'] > 0
        # if self.compression:
        #     self.compressor = NaiveCompressor(256, args['compression'])

        # self.spatial_transform = SpatialTransformation(
        #     args['spatial_transform'])

        self.fusion_net = HeteroFusion(args['hetero_fusion'])

        self.use_hetero_decoder = 'hetero_decoder' in args
        if 'decoder' in args:
            self.decoder = NaiveDecoder(args['decoder'])
        if 'hetero_decoder' in args:
            self.decoder = HeteroDecoder(args['hetero_decoder'])

        self.cls_head = nn.Conv2d(256, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(256, 7 * args['anchor_number'], kernel_size=1)

    def unpad_mode_encoding(self, mode, record_len):
        B = mode.shape[0]
        out = []
        for i in range(B):
            out.append(mode[i, :record_len[i]])
        return torch.cat(out, dim=0)

    def freeze_networks(self, net):
        for p in net.parameters():
            p.requires_grad = False
    
    def inference_freeze(self):
        self.freeze_networks(self)

    def update_model(self, fusion_net, decoder, cls_head, reg_head, dir_head, second_net=None, pillar_net=None, mm_net=None):
        self.fusion_net = fusion_net
        self.decoder = decoder
        self.cls_head = cls_head
        self.reg_head = reg_head
        self.dir_head = dir_head

        if second_net is not None:
            self.second = second_net
            self.freeze_networks(self.second)
            print('second net freezed.')        
        if pillar_net is not None:
            self.pillar = pillar_net
            self.freeze_networks(self.pillar)
            print('pillar net freezed.')
        if mm_net is not None:
            self.basebone = mm_net
            self.freeze_networks(self.basebone)
            print('multimodal net freezed.')

    def regroup(self, x, record_len, select_idx):
        split_x = torch.tensor_split(x, torch.cumsum(record_len, dim=0)[:-1].cpu())
        select_x, select_ego = [], []
        for _x in split_x:
            if _x.shape[0] < select_idx + 1:
                select_x.append([])
            else:
                select_x.append(_x[select_idx:select_idx+1])
            select_ego.append(_x[0:1])
        return select_x, select_ego
        

    def forward(self, data_dict, mode=[0,1], training=False):
        output_dict = {}
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        rec_loss, svd_loss, bfp_loss = torch.tensor(0.0, requires_grad=True).to(record_len.device), torch.tensor(0.0, requires_grad=True).to(record_len.device), torch.tensor(0.0, requires_grad=True).to(record_len.device)

        batch_dict = {
                'voxel_features': data_dict['processed_lidar']['voxel_features'],
                'voxel_coords': data_dict['processed_lidar']['voxel_coords'],
                'voxel_num_points': data_dict['processed_lidar']['voxel_num_points'],
                'image_inputs': data_dict['image_inputs'],
                'batch_size': torch.sum(data_dict['record_len']).cpu().numpy(),
                'record_len': record_len
            }
        features = []
        origin_features = []
        agent_idx = 0

        if self.second is not None:
            f = self.second(data_dict)

            # homo, no action is needed
            if self.type == 1:
                batch_dict['spatial_features'] = f
            
            # only hetero
            else:
                print(f'second {agent_idx} type.')
                if self.feat_proj:
                    f = eval(f"self.a{agent_idx}_proj")({'spatial_features':f})['spatial_features_2d']
                else:
                    print('second no proj.')
                # proj_features.append(f)

                features, _ = self.regroup(f, record_len, select_idx=agent_idx)
                agent_idx += 1



        if self.pillar is not None:
            f = self.pillar(batch_dict)

            # homo
            if self.type == 1:
                batch_dict['spatial_features'] = f

            # hetero
            else:
                print(f'pillar {agent_idx} type.')
                if self.feat_proj:
                    print('pillar proj.')
                    # f = self.minmax_norm(f)
                    f = eval(f"self.a{agent_idx}_proj")({'spatial_features':f})['spatial_features_2d']
                else:
                    print('pillar no proj.')
                # proj_features.append(f)
                
                # when is the 1st agent, no operation is needed
                # when is the 2nd agent
                select_x, _ = self.regroup(f, record_len, select_idx=agent_idx)
                if features:
                    print(f'pillar not first coolaborate agent')
                    for i in range(len(record_len)):
                        if select_x[i] != []:
                            features[i] = torch.cat([features[i], select_x[i]], dim=0)
                    
                    features = torch.cat(features, dim=0)
                    batch_dict['spatial_features'] = features
                
                else:
                    print('pillar first coolaborate agent')
                    features = select_x

                agent_idx += 1
        

        if self.basebone is not None:
            f, rec_loss, svd_loss = self.basebone(batch_dict, mode=mode, training=training)

            # homo, no action is needed
            if self.type == 1:
                batch_dict['spatial_features'] = f

            # only hetero
            else:
                print(f'multi-modal {agent_idx} type.')
                if self.feat_proj:
                    print('multi-modal proj.')
                    # f = self.minmax_norm(f)
                    f = eval(f"self.a{agent_idx}_proj")({'spatial_features':f})['spatial_features_2d']
                else:
                    print('multi-modal no proj.')
                # proj_features.append(f)
                
                # when is the 1st agent, no operation is needed
                # when is the 2nd agent
                select_x, _ = self.regroup(f, record_len, select_idx=agent_idx)
                if features:
                    print('multi-modal not first coolaborate agent')
                    for i in range(len(record_len)):
                        if select_x[i] != []:
                            features[i] = torch.cat([features[i], select_x[i]], dim=0)
                    
                    features = torch.cat(features, dim=0)
                    batch_dict['spatial_features'] = features

                else:
                    print('multi-modal first and last coolaborate agent')
                    batch_dict['spatial_features'] = f
                

                agent_idx += 1


        if self.type > 1 and not (self.pillar is not None and self.second is not None):
            # (B, L)
            # mode = batch['mode'].to(torch.int)
            mode = torch.Tensor([0, 1]).repeat(len(record_len), 1).to(torch.int).to(record_len.device)
        else:
            mode = torch.Tensor([1, 1]).repeat(len(record_len), 1).to(torch.int).to(record_len.device)

        mode_unpack = self.unpad_mode_encoding(mode, record_len)

        ### combine hetero features HAVE DONE
        # camera_features = None
        # lidar_features = None
        # if not torch.all(mode_unpack == 1):
        #     batch_camera = self.extract_camera_input(batch)
        #     camera_features = self.camera_encoder(batch_camera)
        # # If there is at least one lidar
        # if not torch.all(mode_unpack == 0):
        #     batch_lidar = self.extract_lidar_input(batch)
        #     lidar_features = self.lidar_encoder(batch_lidar)
        # x = self.combine_features(camera_features,
        #                           lidar_features, mode_unpack,
        #                           record_len)

        # if self.compression:
        #     x = self.compressor(x)

        # N, C, H, W -> B,  L, C, H, W
        x, mask = regroup(batch_dict['spatial_features'], record_len, self.max_cav)
        # B, L, C, H, W
        x = self.fusion_net(x, pairwise_t_matrix, mode, record_len, mask).squeeze(1)


        if self.use_hetero_decoder:
            psm, rm = self.decoder(x.unsqueeze(1), mode, use_upsample=False)
        else:
            x = self.decoder(x.unsqueeze(1), use_upsample=False).squeeze(1)
            psm = self.cls_head(x)
            rm = self.reg_head(x)
            
        output_dict = {'psm': psm,
                       'rm': rm}

        output_dict.update({
            'cls_preds': psm,
            'reg_preds': rm,
            'rec_loss': rec_loss,
            'svd_loss': svd_loss,
            'bfp_loss': bfp_loss,
        })

        return output_dict


def torch_tensor_to_numpy(torch_tensor):
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    A numpy array.
    """
    return torch_tensor.numpy() if not torch_tensor.is_cuda else \
        torch_tensor.cpu().detach().numpy()


def regroup(dense_feature, record_len, max_len):
    """
    Regroup the data based on the record_len.

    Parameters
    ----------
    dense_feature : torch.Tensor
        N, C, H, W
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
    split_features = torch.tensor_split(dense_feature,
                                        cum_sum_len[:-1])
    regroup_features = []
    mask = []

    for split_feature in split_features:
        # M, C, H, W
        feature_shape = split_feature.shape

        # the maximum M is 5 as most 5 cavs
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)

        padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                     feature_shape[2], feature_shape[3])
        padding_tensor = padding_tensor.to(split_feature.device)

        split_feature = torch.cat([split_feature, padding_tensor],
                                  dim=0)

        # 1, 5C, H, W
        split_feature = split_feature.view(-1,
                                           feature_shape[2],
                                           feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    # B, 5C, H, W
    regroup_features = torch.cat(regroup_features, dim=0)
    # B, L, C, H, W
    regroup_features = rearrange(regroup_features,
                                 'b (l c) h w -> b l c h w',
                                 l=max_len)
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)

    return regroup_features, mask