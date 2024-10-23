# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for late fusion
"""
import random
import math
import torch
import numpy as np
from PIL import Image
import cv2
from collections import OrderedDict

from torch.utils.data import DataLoader

import opencood.data_utils.datasets
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets.opv2v import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils, transformation_utils, camera_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


def merge_features_to_dict(processed_feature_list, merge=None):
    """
    Merge the preprocessed features from different cavs to the same
    dictionary.

    Parameters
    ----------
    processed_feature_list : list
        A list of dictionary containing all processed features from
        different cavs.
    merge : "stack" or "cat". used for images

    Returns
    -------
    merged_feature_dict: dict
        key: feature names, value: list of features.
    """

    merged_feature_dict = OrderedDict()

    for i in range(len(processed_feature_list)):
        for feature_name, feature in processed_feature_list[i].items():

            if feature_name not in merged_feature_dict:
                merged_feature_dict[feature_name] = []
            if isinstance(feature, list):
                merged_feature_dict[feature_name] += feature
            else:
                merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]

    # stack them
    # it usually happens when merging cavs images -> v.shape = [N, Ncam, C, H, W]
    # cat them
    # it usually happens when merging batches cav images -> v is a list [(N1+N2+...Nn, Ncam, C, H, W))]
    if merge=='stack': 
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.stack(features, dim=0)
    elif merge=='cat':
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.cat(features, dim=0)

    return merged_feature_dict


class LateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """
    def __init__(self, params, visualize, train=True):
        super(LateFusionDataset, self).__init__(params, visualize, train)
        if 'f1' in params['preprocess']:
            self.pre_processor = build_preprocessor(params['preprocess']['f1'], train)
            self.pre_processor2 = build_preprocessor(params['preprocess']['f2'], train)
        else:
            self.pre_processor = build_preprocessor(params['preprocess'], train)
            self.pre_processor2 = None
        self.post_processor = build_postprocessor(params['postprocess'], dataset='opv2v', train=train)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict)

        return reformat_data_dict

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = return_timestamp_key(scenario_database, timestamp_index)
        # calculate distance to ego for each cav
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            timestamp_delay = self.time_delay_calculation(cav_content['ego'])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = return_timestamp_key(scenario_database, timestamp_index_delay)

            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag)
            data[cav_id]['lidar_np'] = pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
            img_src = []
            for idx in range(self.data_aug_conf['Ncams']):
                image_path = cav_content[timestamp_key_delay]['camera'][idx]
                img_src.append(Image.open(image_path))
            data[cav_id]['camera_data'] = img_src
        return data

    def reform_param(self, cav_content, ego_content, timestamp_cur, timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """        
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)
        
        if cur_ego_pose_flag:
            transformation_matrix = transformation_utils.x1_to_x2(delay_cav_lidar_pose, cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = transformation_utils.x1_to_x2(delay_cav_lidar_pose, delay_ego_lidar_pose)
            spatial_correction_matrix = transformation_utils.x1_to_x2(delay_ego_lidar_pose, cur_ego_lidar_pose)
        
        # This is only used for late fusion, as it did the transformation in the postprocess, so we want the gt object transformation use the correct one
        gt_transformation_matrix = transformation_utils.x1_to_x2(cur_cav_lidar_pose, cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = cur_params['vehicles']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix
        
        camera_to_lidar_matrix, camera_intrinsic = [], []
        for idx in range(self.data_aug_conf['Ncams']):
            camera_id = self.data_aug_conf['cams'][idx]

            camera_pos = delay_params[camera_id]['cords']
            lidar_pose = delay_params['lidar_pose']
            camera2lidar = transformation_utils.x1_to_x2(camera_pos, lidar_pose)
            extrinsic = delay_params[camera_id]['extrinsic']
            # print(camera2lidar, extrinsic)
            # camera_to_lidar_matrix.append(delay_params[camera_id]['extrinsic'])
            camera_to_lidar_matrix.append(extrinsic)
            camera_intrinsic.append(delay_params[camera_id]['intrinsic'])

        delay_params['camera2lidar_matrix'] = camera_to_lidar_matrix
        delay_params['camera_intrinsic'] = camera_intrinsic
        
        return delay_params

    def get_item_single_car(self, selected_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        # Transformation matrix, intrinsic(cam_K)
        camera_to_lidar_matrix = np.array(selected_cav_base['params']['camera2lidar_matrix']).reshape(4,4,4).astype(np.float32)
        # lidar_to_camera_matrix = np.array(selected_cav_base['params']['lidar2camera_matrix']).astype(np.float32)
        camera_intrinsic = np.array(selected_cav_base['params']["camera_intrinsic"]).reshape(4,3,3).astype(np.float32)

        # data augmentation
        # resize, resize_dims, crop, flip, rotate = camera_utils.sample_augmentation(self.data_aug_conf, self.train)
        # img_src, post_rot2, post_tran2 = camera_utils.img_transform(img_src, torch.eye(2), torch.zeros(2), resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate)
        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(4,3)
        #post_tran[:2] = post_tran2
        post_rot = torch.eye(3).unsqueeze(0).repeat(4,1,1)
        #post_rot[:2, :2] = post_rot2

        # image resize and normalize
        reH, reW = self.data_aug_conf['final_dim'][0], self.data_aug_conf['final_dim'][1]
        img_src = []
        for img in selected_cav_base["camera_data"]:
            imgH, imgW = img.height, img.width
            resized_img = img.resize((reW, reH))
            img_src.append(camera_utils.normalize_img(resized_img))

        selected_cav_processed = {
            "image_inputs":{
                "imgs": torch.stack(img_src, dim=0), # [N(cam), 3, H, W]
                "intrins": torch.from_numpy(camera_intrinsic),  # 4*3*3
                "extrins": torch.from_numpy(camera_to_lidar_matrix),    # 4*3*3
                "rots": torch.from_numpy(camera_to_lidar_matrix[:, :3, :3]),  # R_wc, we consider world-coord is the lidar-coord
                "trans": torch.from_numpy(camera_to_lidar_matrix[:, :3, 3]),  # T_wc
                "post_rots": post_rot,  # 4*3
                "post_trans": post_tran,    # 4*3*3
            }
        }

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)

        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       selected_cav_base[
                                                           'params'][
                                                           'lidar_pose'])
        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = \
            self.augment(lidar_np, object_bbx_center, object_bbx_mask)

        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update({'processed_lidar': lidar_dict})

        if self.pre_processor2 is not None:
            processed_lidar2 = self.pre_processor2.preprocess(lidar_np)
            selected_cav_processed.update({
                'processed_lidar2': processed_lidar2,
            })

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({'anchor_box': anchor_box})

        selected_cav_processed.update({'object_bbx_center': object_bbx_center,
                                       'object_bbx_mask': object_bbx_mask,
                                       'object_ids': object_ids})

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask)
        selected_cav_processed.update({'label_dict': label_dict})

        bev_map = self.get_dynamic_bev_map(selected_cav_processed)
        selected_cav_processed['label_dict']['gt_dynamic'] = bev_map

        return selected_cav_processed

    def get_dynamic_bev_map(self, processed_data_dict):
        bbx_center = processed_data_dict['object_bbx_center']
        bbx_mask = processed_data_dict['object_bbx_mask']
        bbxs = box_utils.boxes_to_corners2d(bbx_center[bbx_mask.astype(bool)], 'hwl')
        lidar_range = self.params['preprocess']['cav_lidar_range']
        resolution = self.params['preprocess']['bev_map_resolution']

        w = round((lidar_range[3] - lidar_range[0]) / resolution)
        h = round((lidar_range[4] - lidar_range[1]) / resolution)
        buf = np.zeros((h, w), dtype=np.uint8)
        bev_map = np.zeros((h, w), dtype=np.uint8)

        for box in bbxs:
            box[:, 0] = (box[:, 0] - lidar_range[0]) / resolution
            box[:, 1] = (box[:, 1] - lidar_range[1]) / resolution
            buf.fill(0)
            cv2.fillPoly(buf, [box[:, :2].round().astype(np.int32)], 1, cv2.INTER_LINEAR)
            bev_map[buf > 0] = 1

        # import matplotlib.pyplot as plt
        # plt.imshow(bev_map)
        # plt.savefig('test_bev.png')
        # plt.close()

        return bev_map

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()

        # during training, we return a random cav's data
        if not self.visualize:
            selected_cav_id, selected_cav_base = random.choice(list(base_data_dict.items()))
        else:
            selected_cav_id, selected_cav_base = list(base_data_dict.items())[0]

        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict):
        processed_data_dict = OrderedDict()
        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0])**2 + (
                                      selected_cav_base['params'][
                                          'lidar_pose'][1] - ego_lidar_pose[
                                          1])**2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            # find the transformation matrix from current cav to ego.
            cav_lidar_pose = selected_cav_base['params']['lidar_pose']
            transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)

            selected_cav_processed = \
                self.get_item_single_car(selected_cav_base)
            selected_cav_processed.update({'transformation_matrix':
                                               transformation_matrix})
            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})

        return processed_data_dict


    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for early and late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        processed_lidar_list2 = []
        image_inputs_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            if 'processed_lidar2' in ego_dict:
                processed_lidar_list2.append(ego_dict['processed_lidar2'])
            image_inputs_list.append(ego_dict['image_inputs'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = self.pre_processor.collate_batch(processed_lidar_list)
        if processed_lidar_list2:
            processed_lidar_torch_dict2 = self.pre_processor.collate_batch(processed_lidar_list2)
        merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='stack')
        
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'processed_lidar2': processed_lidar_torch_dict2,
                                   'image_inputs': merged_image_inputs_dict,
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                transformation_matrix = cav_content['transformation_matrix']
                origin_lidar = [cav_content['origin_lidar']]

                projected_lidar = cav_content['origin_lidar']
                projected_lidar[:, :3] = \
                    box_utils.project_points_by_matrix_torch(
                        projected_lidar[:, :3],
                        transformation_matrix)
                projected_lidar_list.append(projected_lidar)

            # processed lidar dictionary
            processed_lidar_torch_dict = self.pre_processor.collate_batch([cav_content['processed_lidar']])
            merged_image_inputs_dict = merge_features_to_dict([cav_content['image_inputs']], merge='stack')

            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(
                    np.array(cav_content['transformation_matrix'])).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'image_inputs': merged_image_inputs_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch})
            
            if 'processed_lidar2' in cav_content:
                output_dict[cav_id].update({
                    'processed_lidar2': self.pre_processor.collate_batch([cav_content['processed_lidar2']])
                })

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

        if self.visualize:
            projected_lidar_stack = torch.from_numpy(
                np.vstack(projected_lidar_list))
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        # pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        preds = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        # return pred_box_tensor, pred_score, gt_box_tensor
        return preds + (gt_box_tensor,)

    def post_process_no_fusion(self, data_dict, output_dict_ego):
        """
        The object id can not used for identifying the same object.
        here we will to use the IoU to determine it.
        """
        data_dict_ego = OrderedDict()
        data_dict_ego['ego'] = data_dict['ego']
        gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict)

        pred_box_tensor, pred_score, pred_dbev = \
            self.post_processor.post_process(data_dict_ego, output_dict_ego)
        return pred_box_tensor, pred_score, pred_dbev, gt_box_tensor
    