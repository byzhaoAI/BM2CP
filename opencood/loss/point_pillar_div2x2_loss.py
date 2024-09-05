"""
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.loss.point_pillar_loss_for_div2x import PointPillarLoss
from opencood.utils.box_utils import boxes_to_corners2d
from opencood.models.fuse_modules.fusion_in_one import regroup

def calculate_box_mask_gaussian(
    preds_shape, target, pc_range, voxel_size, out_size_scale
):
    B = preds_shape[0]
    C = preds_shape[1]
    H = preds_shape[2]
    W = preds_shape[3]
    gt_mask = np.zeros((B, H, W), dtype=np.float32)  # C * H * W

    for i in range(B):
        for j in range(len(target[i])):
            if target[i][j].sum() == 0:
                break

            w, h = (
                target[i][j][3] / (voxel_size[0] * out_size_scale),
                target[i][j][4] / (voxel_size[1] * out_size_scale),
            )
            radius = gaussian_radius((w, h))
            radius = max(0, int(radius))

            center_heatmap = [
                int((target[i][j][0] - pc_range[0]) / (voxel_size[0] * out_size_scale)),
                int((target[i][j][1] - pc_range[1]) / (voxel_size[1] * out_size_scale)),
            ]
            draw_umich_gaussian(gt_mask[i], center_heatmap, radius)
    return gt_mask


def gaussian_radius(bbox_size, min_overlap=0.7):
    height, width = bbox_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _load_data_to_gpu(data_dict):
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.cuda()
        elif isinstance(v, dict):
            _load_data_to_gpu(data_dict[k])
        else:
            data_dict[k] = v


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

class PointPillarDIV2X2Loss(PointPillarLoss):
    def __init__(self, args):
        super(PointPillarDIV2X2Loss, self).__init__(args)
        self.kd = args['kd']
        print('using PointPillarUniDistillLoss: feature: {}, relation: {}, response: {}, intermediate: {}'.format(self.kd['feature_kd'], self.kd['relation_kd'], self.kd['response_kd'], self.kd['intermediate_kd']))
        self.early_distill = self.kd.get('early_distill', False)
        print('self.early_distill: ', self.early_distill)
        
    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        total_loss = super().forward(output_dict, target_dict)

        # if not self.kd['feature_kd']:
        #     return total_loss

        ########## KL loss ############
        rm = output_dict['reg_preds']  # [B, 14, 50, 176]
        psm = output_dict['cls_preds'] # [B, 2, 50, 176]
        feature = output_dict['feature']

        teacher_rm = output_dict['teacher_reg_preds']
        teacher_psm = output_dict['teacher_cls_preds']

        kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)
        feature = output_dict['feature']
        teacher_feature = output_dict['teacher_feature']
        N, C, H, W = teacher_feature.shape
        teacher_feature = teacher_feature.permute(0,2,3,1).reshape(N*H*W, C)
        student_feature = feature.permute(0,2,3,1).reshape(N*H*W, C)
        kd_loss_feature = kl_loss_mean(
                F.log_softmax(student_feature, dim=1), F.softmax(teacher_feature, dim=1)
            )
        kd_loss = kd_loss_feature * self.kd['weight']

        if self.early_distill != 0:
            single_features, record_len, t_matrix = output_dict['single_features'], output_dict['record_len'], output_dict['t_matrix']
            split_x = regroup(single_features, record_len)
            B, L = t_matrix.shape[:2]
            ego_feature, inf_feature, inf_idx = [], [], []


            for b in range(B):
                cav_num = split_x[b].shape[0]
                ego_feature.append(split_x[b][0])
                if cav_num > 1:
                    inf_feature.append(split_x[b][1])
                    inf_idx.append(b)
            ego_feature = torch.stack(ego_feature) #[B, C, H, W]
            if len(inf_feature) > 0:
                inf_feature = torch.stack(inf_feature) ##[B2, C, H, W]

            #inf_idx = torch.stack(inf_idx)

            teacher_feature = output_dict['teacher_feature']
            N, C, H, W = ego_feature.shape
            teacher_feature = teacher_feature.permute(0,2,3,1).reshape(N*H*W, C)
            ego_feature = ego_feature.permute(0,2,3,1).reshape(N*H*W, C)
            kd_loss_feature = kl_loss_mean(F.log_softmax(ego_feature, dim=1), F.softmax(teacher_feature, dim=1))
            kd_ego_loss = kd_loss_feature * self.early_distill
            kd_loss += kd_ego_loss

            #generate overlap mask
            overlap_mask = torch.ones(B, 1, H, W).to(ego_feature)
            t_matrix_inf = t_matrix[:, 0, 1, :, :]
            grid = F.affine_grid(t_matrix_inf, [B, 1, H, W], align_corners=True).to(ego_feature)
            overlap_mask = F.grid_sample(overlap_mask, grid, align_corners=True)  #[B, 1, H, W]

            if len(inf_feature)> 0 :
                teacher_feature = output_dict['teacher_feature']
                N, C, H, W = inf_feature.shape
                teacher_feature *= overlap_mask #mask
                teacher_feature = teacher_feature[inf_idx]
                teacher_feature = teacher_feature.permute(0,2,3,1).reshape(N*H*W, C)
                inf_feature = inf_feature.permute(0,2,3,1).reshape(N*H*W, C)
                kd_loss_feature = kl_loss_mean(F.log_softmax(inf_feature, dim=1), F.softmax(teacher_feature, dim=1))
                kd_inf_loss = kd_loss_feature * self.early_distill
                kd_loss += kd_inf_loss


        feature = output_dict['feature']
        teacher_feature = output_dict['teacher_feature']
        N, C, H, W = teacher_feature.shape
        gt_boxes= target_dict['object_bbx_center']
        gt_boxes_bev_coords = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 4, 2))
        for i in range(gt_boxes.shape[0]):
            gt_boxes_tmp = gt_boxes[i]
            gt_boxes_tmp_bev = boxes_to_corners2d(gt_boxes_tmp, 'hwl')
            gt_boxes_bev_coords[i] = gt_boxes_tmp_bev[:,:,:2]

        gt_boxes_bev_coords[:, :, :, 0] = (
            gt_boxes_bev_coords[:, :, :, 0] - self.kd['lidar_range'][0]
        ) / (self.kd['voxel_size'][0] * 2)
        gt_boxes_bev_coords[:, :, :, 1] = (
            gt_boxes_bev_coords[:, :, :, 1] - self.kd['lidar_range'][1]
        ) / (self.kd['voxel_size'][1] * 2)

        bev_gt_boxes_mask = (target_dict['object_bbx_mask']==1)
        device = bev_gt_boxes_mask.device
        gt_boxes_bev_coords = gt_boxes_bev_coords.to(device)

        if self.kd.get('intermediate_kd', False):
            single_features, record_len, t_matrix = output_dict['single_features'], output_dict['record_len'], output_dict['t_matrix']
            split_x = regroup(single_features, record_len)
            B, L = t_matrix.shape[:2]
            ego_feature, inf_feature, inf_idx = [], [], []
            for b in range(B):
                cav_num = split_x[b].shape[0]
                ego_feature.append(split_x[b][0])
                if cav_num > 1:
                    inf_feature.append(split_x[b][1])
                    inf_idx.append(b)
            ego_feature = torch.stack(ego_feature) #[B, C, H, W]
            # if len(inf_feature) > 0:
            #     inf_feature = torch.stack(inf_feature) ##[B2, C, H, W]
            #     # generate overlap mask
            #     overlap_mask = torch.ones(B, 1, H, W).to(ego_feature)
            #     t_matrix_inf = t_matrix[:, 0, 1, :, :]
            #     grid = F.affine_grid(t_matrix_inf, [B, 1, H, W], align_corners=True).to(ego_feature)
            #     overlap_mask = F.grid_sample(overlap_mask, grid, align_corners=True)  #[B, 1, H, W]
            # else:
            #     overlap_mask = torch.zeros(B, 1, H, W).to(ego_feature)

            #kd_loss_feature = self.FeatureDistillLoss(ego_feature*(1-overlap_mask), teacher_feature*(1-overlap_mask), gt_boxes_bev_coords, bev_gt_boxes_mask)
            kd_loss_feature = self.FeatureDistillLoss(ego_feature, teacher_feature, gt_boxes_bev_coords, bev_gt_boxes_mask)

            kd_loss += kd_loss_feature * self.kd['intermediate_weight']

            # if len(inf_feature)> 0 :
            #     teacher_feature_inf = teacher_feature[inf_idx]
            #     overlap_mask_inf = overlap_mask[inf_idx]
            #     gt_boxes_bev_coords_inf = gt_boxes_bev_coords[inf_idx]
            #     bev_gt_boxes_mask_inf = bev_gt_boxes_mask[inf_idx]
            #     kd_loss_feature = self.FeatureDistillLoss(inf_feature*(1-overlap_mask_inf), teacher_feature_inf*(1-overlap_mask_inf), gt_boxes_bev_coords_inf, bev_gt_boxes_mask_inf)
            #     #kd_loss_feature = self.FeatureDistillLoss(inf_feature*overlap_mask_inf, teacher_feature_inf*overlap_mask_inf, gt_boxes_bev_coords_inf, bev_gt_boxes_mask_inf)
            #     kd_loss += kd_loss_feature * self.kd['intermediate_weight']

        if self.kd.get('feature_kd', False):
            #kd_loss_feature = self.FeatureDistillLoss(feature*overlap_mask, teacher_feature*overlap_mask, gt_boxes_bev_coords, bev_gt_boxes_mask)
            kd_loss_feature = self.FeatureDistillLoss(feature, teacher_feature, gt_boxes_bev_coords, bev_gt_boxes_mask)
            kd_loss += kd_loss_feature * self.kd['feat_weight']
        if self.kd.get('relation_kd', False):
            kd_loss_relation = self.BEVDistillLoss(feature, teacher_feature, gt_boxes_bev_coords, bev_gt_boxes_mask)
            kd_loss += kd_loss_relation * self.kd['rel_weight']
        if self.kd.get('response_kd', False):
            kd_loss_response = self.ResponseDistillLoss(psm, rm, teacher_psm, teacher_rm, gt_boxes, self.kd['lidar_range'], self.kd['voxel_size'], 2)
            kd_loss += kd_loss_response * self.kd['res_weight']  


        total_loss += kd_loss
        self.loss_dict.update({'total_loss': total_loss.item(),
                              'kd_loss': kd_loss.item()})


        return total_loss


    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=''):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        kd_loss = self.loss_dict.get('kd_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || KD Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, kd_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Kd_loss'+suffix, kd_loss,
                            epoch*batch_len + batch_id)
  

    def FeatureDistillLoss(
        self, feature_lidar, feature_fuse, gt_boxes_bev_coords, gt_boxes_indices
    ):
        
        '''
        feature_lidar: [B, C, H, W]
        gt_boxes_bev_coords: [B, N, 4, 2]
        '''
        h, w = feature_lidar.shape[-2:]
        gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2) #[B, N, 1, 2]
        gt_boxes_bev_edge_1 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_2 = torch.mean(
            gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_3 = torch.mean(
            gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_4 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
        ).unsqueeze(2)
        #[B,N,9,2]
        gt_boxes_bev_all = torch.cat(
            (
                gt_boxes_bev_coords,
                gt_boxes_bev_center,
                gt_boxes_bev_edge_1,
                gt_boxes_bev_edge_2,
                gt_boxes_bev_edge_3,
                gt_boxes_bev_edge_4,
            ),
            dim=2,
        )
        gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
        gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
        gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
        feature_lidar_sample = torch.nn.functional.grid_sample(
            feature_lidar, gt_boxes_bev_all
        )
        feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
        feature_fuse_sample = torch.nn.functional.grid_sample(
            feature_fuse, gt_boxes_bev_all
        )
        feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
        criterion = nn.L1Loss(reduce=False)
        loss_feature_distill = criterion(
            feature_lidar_sample[gt_boxes_indices], feature_fuse_sample[gt_boxes_indices]
        )
        loss_feature_distill = torch.mean(loss_feature_distill, 2)
        loss_feature_distill = torch.mean(loss_feature_distill, 1)
        loss_feature_distill = torch.sum(loss_feature_distill)
        weight = gt_boxes_indices.float().sum()
        #weight = reduce_mean(weight)
        loss_feature_distill = loss_feature_distill / (weight + 1e-4)
        return loss_feature_distill

    def FeatureDistillKDLoss(
        self, feature_lidar, feature_fuse, gt_boxes_bev_coords, gt_boxes_indices
    ):
        
        '''
        feature_lidar: [B, C, H, W]
        gt_boxes_bev_coords: [B, N, 4, 2]
        '''
        h, w = feature_lidar.shape[-2:]
        gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2) #[B, N, 1, 2]
        gt_boxes_bev_edge_1 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_2 = torch.mean(
            gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_3 = torch.mean(
            gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_4 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
        ).unsqueeze(2)
        #[B,N,9,2]
        gt_boxes_bev_all = torch.cat(
            (
                gt_boxes_bev_coords,
                gt_boxes_bev_center,
                gt_boxes_bev_edge_1,
                gt_boxes_bev_edge_2,
                gt_boxes_bev_edge_3,
                gt_boxes_bev_edge_4,
            ),
            dim=2,
        )
        gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
        gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
        gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
        feature_lidar_sample = torch.nn.functional.grid_sample(
            feature_lidar, gt_boxes_bev_all
        )
        feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
        feature_fuse_sample = torch.nn.functional.grid_sample(
            feature_fuse, gt_boxes_bev_all
        )
        feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
        kl_loss_mean = nn.KLDivLoss(reduction='batchmean')
        N, _, _, C = feature_fuse_sample.shape
        teacher_feature = feature_fuse_sample[gt_boxes_indices].reshape(-1, C)
        student_feature = feature_lidar_sample[gt_boxes_indices].reshape(-1, C)
        loss_feature_distill = kl_loss_mean(
                F.log_softmax(student_feature, dim=1), F.softmax(teacher_feature, dim=1)
            )
        # loss_feature_distill = torch.mean(loss_feature_distill, 2)
        # loss_feature_distill = torch.mean(loss_feature_distill, 1)
        # loss_feature_distill = torch.sum(loss_feature_distill)
        # weight = gt_boxes_indices.float().sum()
        # #weight = reduce_mean(weight)
        # loss_feature_distill = loss_feature_distill / (weight + 1e-4)
        return loss_feature_distill


    def BEVDistillLoss(self, bev_lidar, bev_fuse, gt_boxes_bev_coords, gt_boxes_indices):
        h, w = bev_lidar.shape[-2:]
        gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2)
        gt_boxes_bev_edge_1 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_2 = torch.mean(
            gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_3 = torch.mean(
            gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_4 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_all = torch.cat(
            (
                gt_boxes_bev_coords,
                gt_boxes_bev_center,
                gt_boxes_bev_edge_1,
                gt_boxes_bev_edge_2,
                gt_boxes_bev_edge_3,
                gt_boxes_bev_edge_4,
            ),
            dim=2,
        )
        gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
        gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
        gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
        feature_lidar_sample = torch.nn.functional.grid_sample(bev_lidar, gt_boxes_bev_all)
        feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
        feature_fuse_sample = torch.nn.functional.grid_sample(bev_fuse, gt_boxes_bev_all)
        feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
        criterion = nn.L1Loss(reduce=False)
        weight = gt_boxes_indices.float().sum()
        #weight = reduce_mean(weight)
        gt_boxes_sample_lidar_feature = feature_lidar_sample.contiguous().view(
            -1, feature_lidar_sample.shape[-2], feature_lidar_sample.shape[-1]
        )
        gt_boxes_sample_fuse_feature = feature_fuse_sample.contiguous().view(
            -1, feature_fuse_sample.shape[-2], feature_fuse_sample.shape[-1]
        )
        gt_boxes_sample_lidar_feature = gt_boxes_sample_lidar_feature / (
            torch.norm(gt_boxes_sample_lidar_feature, dim=-1, keepdim=True) + 1e-4
        )
        gt_boxes_sample_fuse_feature = gt_boxes_sample_fuse_feature / (
            torch.norm(gt_boxes_sample_fuse_feature, dim=-1, keepdim=True) + 1e-4
        )
        gt_boxes_lidar_rel = torch.bmm(
            gt_boxes_sample_lidar_feature,
            torch.transpose(gt_boxes_sample_lidar_feature, 1, 2),
        )
        gt_boxes_fuse_rel = torch.bmm(
            gt_boxes_sample_fuse_feature,
            torch.transpose(gt_boxes_sample_fuse_feature, 1, 2),
        )
        gt_boxes_lidar_rel = gt_boxes_lidar_rel.contiguous().view(
            gt_boxes_bev_coords.shape[0],
            gt_boxes_bev_coords.shape[1],
            gt_boxes_lidar_rel.shape[-2],
            gt_boxes_lidar_rel.shape[-1],
        )
        gt_boxes_fuse_rel = gt_boxes_fuse_rel.contiguous().view(
            gt_boxes_bev_coords.shape[0],
            gt_boxes_bev_coords.shape[1],
            gt_boxes_fuse_rel.shape[-2],
            gt_boxes_fuse_rel.shape[-1],
        )
        loss_rel = criterion(
            gt_boxes_lidar_rel[gt_boxes_indices], gt_boxes_fuse_rel[gt_boxes_indices]
        )
        loss_rel = torch.mean(loss_rel, 2)
        loss_rel = torch.mean(loss_rel, 1)
        loss_rel = torch.sum(loss_rel)
        loss_rel = loss_rel / (weight + 1e-4)
        return loss_rel


    def ResponseDistillLoss(
        self, cls_s, reg_s, cls_t, reg_t, gt_boxes, pc_range, voxel_size, out_size_scale
    ):
        '''
        cls_s [B, 2, H, W]
        reg_s [B, 14, H, W]
        cls_t [B, 2, H, W]
        reg_t [B, 14, H, W]
        '''

        criterion = nn.L1Loss(reduce=False)

        gaussian_mask = calculate_box_mask_gaussian(
            reg_s.shape,
            gt_boxes.cpu().detach().numpy(),
            pc_range,
            voxel_size,
            out_size_scale,
        )
        device = cls_s.device
        gaussian_mask = torch.from_numpy(gaussian_mask).to(device)
        diff_reg = criterion(reg_s, reg_t)
        diff_cls = criterion(cls_s, cls_t)
        diff_reg = torch.mean(diff_reg, dim=1)
        diff_cls = torch.mean(diff_cls, dim=1)
        diff_reg = diff_reg * gaussian_mask
        diff_cls = diff_cls * gaussian_mask
        weight = gaussian_mask.sum()
        #weight = reduce_mean(weight)
        loss_reg_distill = torch.sum(diff_reg) / (weight + 1e-4)
        loss_cls_distill = torch.sum(diff_cls) / (weight + 1e-4)
        return loss_cls_distill + loss_reg_distill
