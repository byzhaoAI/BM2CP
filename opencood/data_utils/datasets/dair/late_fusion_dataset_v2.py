# -*- coding: utf-8 -*-
# Author: Quanhao Li <quanhaoli2022@163.com> Yifan Lu <yifan_lu@sjtu.edu.cn>, 
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for late fusion
"""
import random
import math
from collections import OrderedDict
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import json

from opencood.data_utils import pre_processor, post_processor, augmentor
from opencood.utils import box_utils, transformation_utils, camera_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import read_json
from opencood.utils.pose_utils import add_noise_data_dict


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


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


class LateFusionDatasetDAIR(Dataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        print(self.visualize)
        self.train = train
        # pre- and post- precessor, data augmentor
        self.data_augmentor = augmentor.data_augmentor.DataAugmentor(params['data_augment'], train)
        if 'f1' in params['preprocess']:
            self.pre_processor = pre_processor.build_preprocessor(params['preprocess']['f1'], train)
            self.pre_processor2 = pre_processor.build_preprocessor(params['preprocess']['f2'], train)
        else:
            self.pre_processor = pre_processor.build_preprocessor(params['preprocess'], train)
            self.pre_processor2 = None
        self.post_processor = post_processor.build_postprocessor(params['postprocess'], dataset='dair', train=train)

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']
        # self.max_cav = 2
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False


        assert 'clip_pc' in params['fusion']['args']
        if params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
        
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None

        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']

        self.split_info = load_json(split_dir)
        co_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.co_data[veh_frame_id] = frame_info

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict, idx)
        
        return reformat_data_dict

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        NOTICE!
        It is different from Intermediate Fusion and Early Fusion

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False
                
        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir, 'vehicle-side/label/lidar/',veh_frame_id + '.json'))
        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))

        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)

        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        # transformation matrix of camera to lidar, and camera intrinsic
        lidar_to_camera_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_camera/'+str(veh_frame_id)+'.json'))
        camera_intrinsic_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/camera_intrinsic/'+str(veh_frame_id)+'.json'))
        data[0]['params']['camera2lidar_matrix'] = np.linalg.inv(transformation_utils.rot_and_trans_to_trasnformation_matrix(lidar_to_camera_json_file))
        data[0]['params']['camera_intrinsic'] = camera_intrinsic_json_file['cam_K']

        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
        data[0]['camera_data'] = Image.open(os.path.join(self.root_dir,frame_info['vehicle_image_path']))

        data[1]['params'] = OrderedDict()
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")

        data[1]['params']['vehicles'] = load_json(os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/',inf_frame_id + '.json'))
        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix1)

        # transformation matrix of camera to lidar, and camera intrinsic
        vlidar_to_camera_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_camera/'+str(inf_frame_id)+'.json'))
        camera_intrinsic_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/camera_intrinsic/'+str(inf_frame_id)+'.json'))
        data[1]['params']['camera2lidar_matrix'] = np.linalg.inv(transformation_utils.rot_and_trans_to_trasnformation_matrix(vlidar_to_camera_json_file))
        data[1]['params']['camera_intrinsic'] = camera_intrinsic_json_file['cam_K']

        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))
        data[1]['camera_data'] = Image.open(os.path.join(self.root_dir,frame_info['infrastructure_image_path']))

        return data

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
        camera_to_lidar_matrix = np.array(selected_cav_base['params']['camera2lidar_matrix']).astype(np.float32)
        # lidar_to_camera_matrix = np.array(selected_cav_base['params']['lidar2camera_matrix']).astype(np.float32)
        camera_intrinsic = np.array(selected_cav_base['params']["camera_intrinsic"]).reshape(3,3).astype(np.float32)
        # image and depth
        imgH, imgW = selected_cav_base["camera_data"].height, selected_cav_base["camera_data"].width
        img_src = [selected_cav_base["camera_data"]]

        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)

        resized_src = []
        for img in img_src:
            reH, reW = self.data_aug_conf['final_dim'][0], self.data_aug_conf['final_dim'][1]
            _img = img.resize((reW, reH))
            _img = camera_utils.normalize_img(_img)
            resized_src.append(_img)
        img_src = resized_src

        selected_cav_processed = {
            "image_inputs":{
                "imgs": torch.stack(img_src, dim=0), # [N(cam), 3or4, H, W]
                "intrins": torch.from_numpy(camera_intrinsic).unsqueeze(0),
                "extrins": torch.from_numpy(camera_to_lidar_matrix).unsqueeze(0),
                "rots": torch.from_numpy(camera_to_lidar_matrix[:3, :3]).unsqueeze(0),  # R_wc, we consider world-coord is the lidar-coord
                "trans": torch.from_numpy(camera_to_lidar_matrix[:3, 3]).unsqueeze(0),  # T_wc
                "post_rots": post_rot.unsqueeze(0),
                "post_trans": post_tran.unsqueeze(0),
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
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                    selected_cav_base[
                                                           'params'][
                                                           'lidar_pose_clean'])

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

    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list, no use.
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_dairv2x_late_fusion(cav_contents[0]) 
        
    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()
        base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])
        # during training, we return a random cav's data
        # only one vehicle is in processed_data_dict
        if not self.visualize:
            selected_cav_id, selected_cav_base = random.choice(list(base_data_dict.items()))
        else:
            selected_cav_id, selected_cav_base = list(base_data_dict.items())[0]
        
        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict, idx):
        """
            processed_data_dict.keys() = ['ego', "650", "659", ...]
        """
        base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        ego_id = -1
        ego_lidar_pose = []
        cav_id_list = []
        lidar_pose_list = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            # if distance > self.params['comm_range']:
            #     continue
            cav_id_list.append(cav_id)
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])

        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]
            # find the transformation matrix from current cav to ego.
            cav_lidar_pose = selected_cav_base['params']['lidar_pose']
            transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
            cav_lidar_pose_clean = selected_cav_base['params']['lidar_pose_clean']
            transformation_matrix_clean = x1_to_x2(cav_lidar_pose_clean, ego_lidar_pose_clean)

            selected_cav_processed = \
                self.get_item_single_car(selected_cav_base)
            selected_cav_processed.update({'transformation_matrix': transformation_matrix,
                                           'transformation_matrix_clean': transformation_matrix_clean})
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

        record_len = []
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
            record_len.append(1)
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
                                   'anchor_box': torch.from_numpy(ego_dict['anchor_box']),
                                   'label_dict': label_torch_dict})

        output_dict['ego'].update({
            'record_len': torch.from_numpy(np.array(record_len, dtype=int)),
        })

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

                if (self.params['only_vis_ego'] is False) or (cav_id=='ego'):
                    import copy
                    projected_lidar = copy.deepcopy(cav_content['origin_lidar'])
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
            
            # late fusion training, no noise
            transformation_matrix_clean_torch = transformation_matrix_torch

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'image_inputs': merged_image_inputs_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch,
                                        'transformation_matrix_clean': transformation_matrix_clean_torch})

            output_dict[cav_id].update({
                'record_len': torch.from_numpy(np.array([1], dtype=int)),
            })
            
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
            projected_lidar_stack = [torch.from_numpy(
                np.vstack(projected_lidar_list))]
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})

            output_dict['ego'].update({'origin_lidar_v':
                    [torch.from_numpy(projected_lidar_list[0])]})
            output_dict['ego'].update({'origin_lidar_i':
                    [torch.from_numpy(projected_lidar_list[1])]})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        
        The object id can not used for identifying the same object.
        here we will to use the IoU to determine it.

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
        gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict)

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
    
    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask