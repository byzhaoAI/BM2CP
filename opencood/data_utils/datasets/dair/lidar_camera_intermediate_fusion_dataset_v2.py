# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
"""
hybrid lidar and camera dataset
"""
import math
from copy import deepcopy
# import time
# from concurrent.futures import process
import os
import numpy as np
import json
import torch
import cv2
import bisect
from PIL import Image
from collections import OrderedDict

from decimal import Decimal

from opencood.data_utils import pre_processor, post_processor, augmentor
from opencood.hypes_yaml import yaml_utils
from opencood.utils import pcd_utils, box_utils, pose_utils, transformation_utils, camera_utils

import matplotlib.pyplot as plt
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d


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


class LiDARCameraIntermediateFusionDatasetDAIR(torch.utils.data.Dataset):
    """
    This class is for intermediate fusion where each vehicle transmit the detection outputs to ego.
    """
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.max_cav = 2
        
        # configs in yaml file about project first, knowledge distillation
        #       if project first, cav's lidar will first be projected to the ego's coordinate frame. otherwise, the feature will be projected instead.
        #       if clip_pc, then clips the lower bound of x-coordinate in point cloud data to 0
        assert 'proj_first' in params['fusion']['args']
        assert 'clip_pc' in params['fusion']['args']

        self.proj_first = True if params['fusion']['args']['proj_first'] else False
        self.kd_flag = params['kd_flag'] if "kd_flag" in params.keys() else False
        self.clip_pc = True if params['fusion']['args']['clip_pc'] else False
        # self.select_keypoint = params['select_kp'] if 'select_kp' in params else None
        self.fg_mask = params['fg_mask'] if 'fg_mask' in params else False
        self.use_gt_depth = True if ('camera_params' in params and params['camera_params']['use_depth_gt']) else False
        self.use_fg_mask = True if ('use_fg_mask' in params['loss']['args'] and params['loss']['args']['use_fg_mask']) else False
        # intermediate and supervise single
        self.supervise_single = True if ('supervise_single' in params['train_params'] and params['train_params']['supervise_single']) else False

        # pre- and post- precessor, data augmentor
        self.grid_conf = params["fusion"]["args"]["grid_conf"]
        self.depth_discre = camera_utils.depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode'])
        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]
        self.pre_processor = pre_processor.build_preprocessor(params['preprocess'], train)
        self.post_processor = post_processor.build_postprocessor(params['postprocess'], train)
        self.data_augmentor = augmentor.data_augmentor.DataAugmentor(params['data_augment'], train)

        # load dataset json file (info) and data path
        self.root_dir = params['data_dir']
        print("Dataset dir:", self.root_dir)

        # slect to choose training set or validation set
        split_dir = params['root_dir'] if self.train else params['validate_dir']
        self.split_info = load_json(split_dir)
        print('Dataset length: {}'.format(len(self.split_info)))

        # save cooperative infra-vehicle pair path&offset data info
        self.coop_data = OrderedDict()
        self.coop_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.veh_datainfo = load_json(os.path.join(self.root_dir, 'vehicle-side/data_info.json'))
        self.inf_datainfo = load_json(os.path.join(self.root_dir, 'infrastructure-side/data_info.json'))
        
        veh_image_path_list = [x['image_path'].split("/")[-1].replace(".jpg", "") for x in self.veh_datainfo]
        inf_image_path_list = [x['image_path'].split("/")[-1].replace(".jpg", "") for x in self.inf_datainfo]
        for clip_idx, clip_info in enumerate(self.coop_datainfo):
            # vehicle image id = cooperative label id
            veh_clip_id = clip_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            inf_clip_id = clip_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
            try:
                veh_index = veh_image_path_list.index(veh_clip_id)
                inf_index = inf_image_path_list.index(inf_clip_id)
            except ValueError:
                continue
            self.coop_data[veh_clip_id] = [clip_idx, veh_index, inf_index]

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        base_data_dict = pose_utils.add_noise_data_dict(base_data_dict,self.params['noise_setting'])

        cav_id_list = []
        lidar_pose_list = []
        lidar_pose_clean_list = []

        agents_image_inputs = []
        processed_features = []
        projected_lidar_clean_list = []

        image_timestamps = []
        lidar_timestamps = []
        
        object_id_stack = []
        object_id_stack_single_v = []
        object_id_stack_single_i = []
        object_stack = []
        object_stack_single_v = []
        object_stack_single_i = []
        single_label_list = []
        
        if self.visualize:
            projected_lidar_stack = []

        # filtered cav within the communication range, then we can calculate get_pairwise_transformation
        # too_far = []
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego, sqrt(x**2+y**2)
            # distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + 
            #                      (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)

            # if distance is too far, we will just skip this agent
            # if distance > self.params['comm_range']:
            #     too_far.append(cav_id)
            #     continue

            cav_id_list.append(cav_id)
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])

        # for cav_id in too_far:
        #     base_data_dict.pop(cav_id)
        
        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]

        pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.max_cav)
        
        # loop over all CAVs to process information
        for cav_id in cav_id_list:
            # structure of selected_cav_processed:
            # {
            #     'image_inputs':{"imgs":, "intrins":, "extrins":, "rots":, "trans":, "post_rots":, "post_trans":}
            #     'object_bbx_center':, 'object_bbx_mask':,
            #     #'object_bbx_center_single':, 'object_ids_single':,
            #     'object_ids':, 'projected_lidar':, 'processed_features':,
            #     'transformation_matrix':, 'transformation_matrix_clean':,
            #     'projected_lidar_clean': (optional for self.kd)
            # }
            selected_cav_processed = self.get_item_single_car(base_data_dict[cav_id], base_data_dict[0])

            
            object_id_stack += selected_cav_processed['object_ids']
            object_stack.append(selected_cav_processed['object_bbx_center'])
            if cav_id == 0: # ego vehicle load cooperative label (id&bbx)
                object_id_stack_single_v += selected_cav_processed['object_ids_single']
                object_stack_single_v.append(selected_cav_processed['object_bbx_center_single'])
            else:           # infrastructure vehicle load local label (id&bbx)
                object_id_stack_single_i += selected_cav_processed['object_ids_single']
                object_stack_single_i.append(selected_cav_processed['object_bbx_center_single'])

            processed_features.append(selected_cav_processed['processed_features'])
            agents_image_inputs.append(selected_cav_processed['image_inputs'])

            # collect timestamps
            lidar_timestamps.append(base_data_dict[cav_id]['params']['lidar_timestamp'])
            image_timestamps.append(base_data_dict[cav_id]['params']['camera_timestamp'])

            if self.kd_flag:
                projected_lidar_clean_list.append(selected_cav_processed['projected_lidar_clean'])

            if self.visualize:
                projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
      
            if self.supervise_single:
                single_label_list.append(selected_cav_processed['single_label_dict'])
        
        # merge preprocessed features from different cavs into the same dict
        merged_feature_dict = merge_features_to_dict(processed_features)
        merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
        if self.kd_flag:    # for disconet knowledge distillation
            stack_lidar_np = np.vstack(projected_lidar_clean_list)
            stack_lidar_np = pcd_utils.mask_points_by_range(stack_lidar_np, self.params['preprocess']['cav_lidar_range'])
            stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
        
        # exclude all repetitive objects    
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]
        # make sure bounding boxes across all frames have the same number
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        object_bbx_center_single_v, mask_single_v, object_id_stack_single_v = self.get_unique_label(object_stack_single_v, object_id_stack_single_v)
        object_bbx_center_single_i, mask_single_i, object_id_stack_single_i = self.get_unique_label(object_stack_single_i, object_id_stack_single_i)

        # generate the anchor boxes and generate targets label
        anchor_box = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center,anchors=anchor_box,mask=mask)
        label_dict_single_v = self.post_processor.generate_label(gt_box_center=object_bbx_center_single_v, anchors=anchor_box, mask=mask_single_v)
        label_dict_single_i = self.post_processor.generate_label(gt_box_center=object_bbx_center_single_i, anchors=anchor_box, mask=mask_single_i)
        
        # data dict per batch
        processed_data_dict = {
            'ego':{
                'sample_idx': idx, 
                'cav_id_list': cav_id_list,
                'cav_num': len(agents_image_inputs),
                'lidar_poses': lidar_poses,
                'lidar_poses_clean': lidar_poses_clean,
                'pairwise_t_matrix': pairwise_t_matrix,
                'image_inputs': merged_image_inputs_dict,
                'processed_lidar': merged_feature_dict,
                
                'image_timestamps': image_timestamps,
                'lidar_timestamps': lidar_timestamps,
                
                'label_dict': label_dict,
                'label_dict_single_v': label_dict_single_v,
                'label_dict_single_i': label_dict_single_i,

                'anchor_box': anchor_box,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,

                'object_ids_single_v': object_id_stack_single_v,
                'object_bbx_center_single_v': object_bbx_center_single_v,
                'object_bbx_mask_single_v': mask_single_v,
                'object_ids_single_i': object_id_stack_single_i,
                'object_bbx_center_single_i': object_bbx_center_single_i,
                'object_bbx_mask_single_i': mask_single_i,
            }
        }
        group = processed_data_dict['ego']['image_inputs']
        #print(group['imgs'].shape, group['intrins'].shape, group['rots'].shape, group['trans'].shape, group['post_rots'].shape, group['post_trans'].shape)

        if self.supervise_single:   # generate single view GT label
            single_label_dicts = self.post_processor.collate_batch(single_label_list)
            processed_data_dict['ego'].update({"single_label_dict_torch": single_label_dicts})

        if self.kd_flag:
            processed_data_dict['ego'].update({'teacher_processed_lidar': stack_feature_processed})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar': np.vstack(projected_lidar_stack)})
            processed_data_dict['ego'].update({'origin_lidar_v': projected_lidar_stack[0]})
            processed_data_dict['ego'].update({'origin_lidar_i': projected_lidar_stack[1]})
        return processed_data_dict

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        
        structure: {
            0: {                    # for vehicle
                'ego':{True}, 
                'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_timestamp':,
                           'camera_timestamp':, 'camera2lidar_matrix':, 'camera_intrinsic':
                }, 
                'lidar_np':, 
                'camera_data':
            },  
            1: {                    # for infrastructure
                'ego':{False}, 
                'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_timestamp':,
                           'camera_timestamp':, 'camera2lidar_matrix':, 'camera_intrinsic':
                }, 
                'lidar_np':, 
                'camera_data':
            }  
        }
        """
        # veh_clip_id: frame id of vehicle in train.json. e.g. "000010", "000011", "000013"
        # index_group: coopration and individual info of infrastructure&vehicle. 
        #       e.g. {"infrastructure_image_path": ..., 
        #             "infrastructure_pointcloud_path": ...,
        #             ...,
        #             "system_error_offset": {"delta_x": ..., "delta_y": ...}}
        veh_clip_id = self.split_info[idx]
        clip_idx, veh_idx, inf_idx = self.coop_data[veh_clip_id]
        system_error_offset = self.coop_datainfo[clip_idx]["system_error_offset"]

        # build dictionary for each agent
        #       e.g. {0: {'ego':{True}, 'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_timestamp':, 'camera_timestamp':, 'camera2lidar_matrix':, 'camera_intrinsic':}, 'lidar_np':, 'camera_data':},  # for vehicle
        #             1: {'ego':{False}, 'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_timestamp':, 'camera_timestamp':, 'camera2lidar_matrix':, 'camera_intrinsic':}, 'lidar_np':, 'camera_data':}  # for infrastructure
        #            }
        data = OrderedDict()
        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[0]['params'] = OrderedDict()
        data[1] = OrderedDict()
        data[1]['ego'] = False
        data[1]['params'] = OrderedDict()


        # vehicle-side
        # load cooperative label once in vehicle side
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir, self.coop_datainfo[clip_idx]['cooperative_label_path']))
        # 6-DOF pose, coordinate system: lidar -> novatel -> world
        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side', self.veh_datainfo[veh_idx]['calib_lidar_to_novatel_path']))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side', self.veh_datainfo[veh_idx]['calib_novatel_to_world_path']))
        transformation_matrix = transformation_utils.veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file, novatel_to_world_json_file)
        data[0]['params']['lidar_pose'] = transformation_utils.tfm_to_pose(transformation_matrix)
        # transformation matrix of camera to lidar, and camera intrinsic
        lidar_to_camera_json_file = load_json(os.path.join(self.root_dir,'vehicle-side', self.veh_datainfo[veh_idx]['calib_lidar_to_camera_path']))
        camera_intrinsic_json_file = load_json(os.path.join(self.root_dir,'vehicle-side', self.veh_datainfo[veh_idx]['calib_camera_intrinsic_path']))
        data[0]['params']['camera2lidar_matrix'] = np.linalg.inv(transformation_utils.rot_and_trans_to_trasnformation_matrix(lidar_to_camera_json_file))
        data[0]['params']['camera_intrinsic'] = camera_intrinsic_json_file['cam_K']
        # label in single view
        data[0]['params']['vehicles_single'] = load_json(os.path.join(self.root_dir, 'vehicle-side', self.veh_datainfo[veh_idx]['label_lidar_std_path']))
        # timestamp&data of lidar&image
        data[0]['params']['lidar_timestamp'] = int(self.veh_datainfo[veh_idx]['pointcloud_timestamp'])
        data[0]['params']['camera_timestamp'] = int(self.veh_datainfo[veh_idx]['image_timestamp'])
        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,self.coop_datainfo[clip_idx]['vehicle_pointcloud_path']))
        data[0]['camera_data'] = deepcopy(Image.open(os.path.join(self.root_dir,self.coop_datainfo[clip_idx]['vehicle_image_path'])))
        if self.clip_pc:
            data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:,0] > 0]

        # infrastructure-side
        # only load cooperative label once in vehicle side
        data[1]['params']['vehicles'] = []
        # 6-DOF pose, coordinate system: lidar -> world
        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side', self.inf_datainfo[inf_idx]['calib_virtuallidar_to_world_path']))
        transformation_matrix1 = transformation_utils.inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file, system_error_offset)
        data[1]['params']['lidar_pose'] = transformation_utils.tfm_to_pose(transformation_matrix1)
        # transformation matrix of camera to lidar, and camera intrinsic
        vlidar_to_camera_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side', self.inf_datainfo[inf_idx]['calib_virtuallidar_to_camera_path']))
        camera_intrinsic_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side', self.inf_datainfo[inf_idx]['calib_camera_intrinsic_path']))
        data[1]['params']['camera2lidar_matrix'] = np.linalg.inv(transformation_utils.rot_and_trans_to_trasnformation_matrix(vlidar_to_camera_json_file))
        data[1]['params']['camera_intrinsic'] = camera_intrinsic_json_file['cam_K']
        # label in single view
        data[1]['params']['vehicles_single'] = load_json(os.path.join(self.root_dir, 'infrastructure-side', self.inf_datainfo[inf_idx]['label_lidar_std_path']))
        # timestamp&data of lidar&image
        data[1]['params']['lidar_timestamp'] = int(self.inf_datainfo[inf_idx]['pointcloud_timestamp'])
        data[1]['params']['camera_timestamp'] = int(self.inf_datainfo[inf_idx]['image_timestamp'])
        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,self.coop_datainfo[clip_idx]["infrastructure_pointcloud_path"]))
        data[1]['camera_data'] = deepcopy(Image.open(os.path.join(self.root_dir,self.coop_datainfo[clip_idx]['infrastructure_image_path'])))        

        return data

    def get_item_single_car(self, selected_cav_base, ego_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
            including 'params', 'camera_data'
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        ego_pose_clean : list, length 6
            only used for gt box generation

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        # process camera data, adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py
        
        # Transformation matrix, intrinsic(cam_K)
        camera_to_lidar_matrix = np.array(selected_cav_base['params']['camera2lidar_matrix']).astype(np.float32)
        # lidar_to_camera_matrix = np.array(selected_cav_base['params']['lidar2camera_matrix']).astype(np.float32)
        camera_intrinsic = np.array(selected_cav_base['params']["camera_intrinsic"]).reshape(3,3).astype(np.float32)

        # image and depth
        imgH, imgW = selected_cav_base["camera_data"].height, selected_cav_base["camera_data"].width
        ori_src = [selected_cav_base["camera_data"]]
        ori_src = [torch.tensor(np.array(src)) for src in ori_src]
        img_src = [selected_cav_base["camera_data"]]
        if self.use_gt_depth:
            img_src.append(selected_cav_base["depth_data"])

        if self.use_fg_mask:
            _, _, fg_mask = camera_utils.coord_3d_to_2d(box_utils.boxes_to_corners_3d(object_bbx_center[:len(object_ids)], self.params['postprocess']['order']),camera_intrinsic,camera_to_lidar_matrix) 
            img_src.append(Image.fromarray(np.array(fg_mask*255, dtype=np.uint8)))

        # data augmentation
        # resize, resize_dims, crop, flip, rotate = camera_utils.sample_augmentation(self.data_aug_conf, self.train)
        # img_src, post_rot2, post_tran2 = camera_utils.img_transform(img_src, torch.eye(2), torch.zeros(2), resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate)
        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        #post_tran[:2] = post_tran2
        post_rot = torch.eye(3)
        #post_rot[:2, :2] = post_rot2

        # image resize
        resized_src = []
        for img in img_src:
            reH, reW = self.data_aug_conf['final_dim'][0], self.data_aug_conf['final_dim'][1]
            resized_src.append(img.resize((reW, reH)))
        img_src = resized_src

        # decouple RGB and Depth
        img_src[0] = camera_utils.normalize_img(img_src[0])
        if self.use_gt_depth:
            img_src[1] = camera_utils.img_to_tensor(img_src[1]) * 255
        if self.use_fg_mask:
            img_src[-1] = camera_utils.img_to_tensor(img_src[-1])

        selected_cav_processed = {
            "image_inputs":{
                "ori_imgs": torch.cat(ori_src, dim=0).unsqueeze(0),
                "imgs": torch.cat(img_src, dim=0).unsqueeze(0), # [Ncam, 3or4, H, W]
                "intrins": torch.from_numpy(camera_intrinsic).unsqueeze(0),
                "extrins": torch.from_numpy(camera_to_lidar_matrix).unsqueeze(0),
                "rots": torch.from_numpy(camera_to_lidar_matrix[:3, :3]).unsqueeze(0),  # R_wc, we consider world-coord is the lidar-coord
                "trans": torch.from_numpy(camera_to_lidar_matrix[:3, 3]).unsqueeze(0),  # T_wc
                "post_rots": post_rot.unsqueeze(0),
                "post_trans": post_tran.unsqueeze(0),
            }
        }
        #group = selected_cav_processed['image_inputs']
        #print(group['imgs'].shape, group['intrins'].shape, group['rots'].shape, group['trans'].shape, group['post_rots'].shape, group['post_trans'].shape)

        # process lidar data

        # calculate the transformation matrix
        transformation_matrix = transformation_utils.x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose']) # T_ego_cav
        transformation_matrix_clean = transformation_utils.x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_cav_base['params']['lidar_pose_clean'])

        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center_dairv2x(selected_cav_base, selected_cav_base['params']['lidar_pose'])
        object_bbx_center_single, object_bbx_mask_single, object_ids_single = self.post_processor.generate_object_center_dairv2x_late_fusion(selected_cav_base)

        # anchor box and then generate targets label single GT
        # anchor_box = self.post_processor.generate_anchor_box()
        # visibility_map = np.asarray(cv2.cvtColor(ego_cav_base["bev_visibility.png"], cv2.COLOR_BGR2GRAY))
        # object_bbx_center_single, object_bbx_mask_single, object_ids_single = self.post_processor.generate_object_center_dairv2x(selected_cav_base, selected_cav_base['params']['lidar_pose'], visibility_map)
        # label_dict_single = self.post_processor.generate_label(gt_box_center=object_bbx_center_single, anchors=anchor_box, mask=object_bbx_mask_single)
        # selected_cav_processed.update({"single_label_dict": label_dict_single})

        lidar_np = selected_cav_base['lidar_np']
        lidar_np = pcd_utils.shuffle_points(lidar_np)
        # Remove the lidar points of the ego vehicle itself.
        lidar_np = pcd_utils.mask_ego_points(lidar_np)
        # project the lidar to ego space, x,y,z in ego space
        projected_lidar = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)

        if self.proj_first:
            lidar_np[:, :3] = projected_lidar

        # filter points out of requirement range
        lidar_np = pcd_utils.mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])

        # point cloud -> voxel feature
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # ============================================================================================================
        # project pcs to img view, create depth map for self
        xyz, int_matrix, ext_matrix = lidar_np[:,:3], camera_intrinsic, camera_to_lidar_matrix
        #if self.left_hand:  # left_hand = True if "OPV2V" in hypes['test_dir'] else False
        #    xyz[:,1] = - xyz[:,1]
        depth_map = self.generate_depth_map(resized_src[0], xyz, int_matrix, ext_matrix, imgH, imgW, draws=False)
        # ============================================================================================================
        # create depth map for ego
        
        xyz_for_ego = projected_lidar[:,:3]
        int_matrix_for_ego = np.array(ego_cav_base['params']["camera_intrinsic"]).reshape(3,3).astype(np.float32)
        ext_matrix_for_ego = np.array(ego_cav_base['params']['camera2lidar_matrix']).astype(np.float32)
        depth_map_for_ego = self.generate_depth_map(resized_src[0], xyz_for_ego, int_matrix_for_ego, ext_matrix_for_ego, imgH, imgW, draws=False)
        depth_map = torch.cat([depth_map.unsqueeze(0), depth_map_for_ego.unsqueeze(0)], dim=0)  # torch.Size([2, 1, 360, 480])
        
        # ============================================================================================================
        selected_cav_processed["image_inputs"].update({"depth_map": depth_map})

        # ============================================================================================================
        """
        plt.imshow(selected_cav_processed["image_inputs"]["depth_map"][0].numpy().transpose(1,2,0))
        plt.savefig('./depth.png')
        plt.close()
        
        canvas = canvas_3d.Canvas_3D(canvas_shape=(imgH, imgW), left_hand=False)
        canvas_xy, valid_mask = canvas.get_canvas_coords(lidar_np)
        canvas.draw_canvas_points(canvas_xy[valid_mask])
        plt.imshow(canvas.canvas)
        plt.savefig('./canvas.png', transparent=False, dpi=400)
        plt.close()
        """
        # ============================================================================================================

        # update dictionary
        selected_cav_processed.update(
            {
                'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
                'object_bbx_mask': object_bbx_mask,
                'object_bbx_center_single': object_bbx_center_single[object_bbx_mask_single == 1],
                'object_ids_single': object_ids_single,
                "object_ids": object_ids,
                'projected_lidar': projected_lidar,
                'processed_features': processed_lidar,
                'transformation_matrix': transformation_matrix,
                'transformation_matrix_clean': transformation_matrix_clean
            }
        )

        if self.kd_flag:
            lidar_np_clean = pcd_utils.mask_ego_points(selected_cav_base['lidar_np']) # copy.deepcopy(lidar_np)
            projected_lidar_clean = box_utils.project_points_by_matrix_torch(lidar_np_clean[:, :3], transformation_matrix_clean)
            lidar_np_clean[:, :3] = projected_lidar_clean
            lidar_np_clean = pcd_utils.mask_points_by_range(lidar_np_clean, self.params['preprocess']['cav_lidar_range'])
            selected_cav_processed.update({"projected_lidar_clean": lidar_np_clean})

        return selected_cav_processed

    def generate_depth_map(self, image, xyz, int_matrix, ext_matrix, imgH, imgW, draws=True):
        xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)   # (..., 3) -> (..., 4)[xyz+1]
        
        if draws:
            print('xyz maxmin: ', xyz_hom.max(axis=0), xyz_hom.min(axis=0))

            plt.imshow(np.transpose(image, (1,2,0)))
            plt.savefig('image.png')
            plt.close()
            
            canvas = canvas_3d.Canvas_3D(canvas_shape=(imgH, imgW), left_hand=False)
            canvas_xy, valid_mask = canvas.get_canvas_coords(xyz_hom)
            canvas.draw_canvas_points(canvas_xy[valid_mask])
            plt.imshow(canvas.canvas)
            plt.savefig('canvas_3d.png', transparent=False, dpi=400)
            plt.close()
            
            camera_pts = (np.linalg.inv(ext_matrix)[:3,:4] @ xyz_hom.T).transpose(1,0)
            camera_pts[:,1] = -camera_pts[:,1]
            canvas = canvas_3d.Canvas_3D(canvas_shape=(3000, 3000), left_hand=False)
            canvas_xy, valid_mask = canvas.get_canvas_coords(camera_pts)
            canvas.draw_canvas_points(canvas_xy[valid_mask])
            plt.imshow(np.transpose(canvas.canvas, (1,0,2)))
            plt.savefig('canvas_cameraview.png', transparent=False, dpi=400)
            plt.close()
            
        ext_matrix = np.linalg.inv(ext_matrix)[:3,:4]
        img_pts = (int_matrix @ ext_matrix @ xyz_hom.T).T

        depth = img_pts[:, 2]
        #print('depth: ', np.max(depth), np.min(depth))
        uv = img_pts[:, :2] / depth[:, None]
        # uv_int = uv.round().astype(np.int32)
        uv_int = (np.ceil(uv) - ((uv - np.floor(uv)) < 0.5).astype(np.int32)).astype(np.int32)
        uv_int = uv_int[:, ::-1]

        valid_mask = ((depth >= self.grid_conf['ddiscr'][0]) &
                      (uv_int[:,0] >= 0) & (uv_int[:,0] < imgH) & 
                      (uv_int[:,1] >= 0) & (uv_int[:,1] < imgW))
        valid_uvint, valid_depth = uv_int[valid_mask], depth[valid_mask]

        depth_map = -1 * np.ones((imgH, imgW), dtype=np.float32)
        for idx, valid_coord in enumerate(valid_uvint):
            u, v = valid_coord[0], valid_coord[1]
            depth_level = bisect.bisect_left(self.depth_discre, valid_depth[idx])
            if depth_level == 0:
                depth_level == 1 
            depth_map[u,v] = depth_level-1 if depth_map[u,v]<0 else min(depth_map[u,v], depth_level-1)
            #depth_map[u,v] = valid_depth[idx] if depth_map[u,v]<0 else min(depth_map[u,v], valid_depth[idx])

        # resize as config requires
        assert imgH % self.data_aug_conf['final_dim'][0] == 0
        assert imgW % self.data_aug_conf['final_dim'][1] == 0
        scaleH, scaleW = imgH // self.data_aug_conf['final_dim'][0], imgW // self.data_aug_conf['final_dim'][1]

        max_depth_level = np.max(depth_map)
        depth_map[depth_map<0] = max_depth_level + 1
        depth_map = torch.FloatTensor(-1 * depth_map).unsqueeze(0)
        pool_layer = torch.nn.MaxPool2d(kernel_size=(scaleH, scaleW), stride=(scaleH, scaleW))
        depth_map = -1 * pool_layer(depth_map)
        depth_map[depth_map>max_depth_level] = -1

        if draws:
            plt.imshow(depth_map.numpy().transpose(1,2,0))
            plt.savefig('depth.png')
            plt.close()
        return depth_map

    def get_unique_label(self, object_stack, object_id_stack):
        # IoU
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        if len(object_stack) > 0:
            # exclude all repetitive objects
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack) if len(object_stack) > 1 else object_stack[0]
            object_stack = object_stack[unique_indices]
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            updated_object_id_stack = [object_id_stack[i] for i in unique_indices]
        else:
            updated_object_id_stack = object_id_stack
        return object_bbx_center, mask, updated_object_id_stack   
    
    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4), L is the max cav number in a scene
            pairwise_t_matrix[i, j] is Tji, i_to_j
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix becomes identity. no need to warp again in fusion time.
            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                lidar_pose = cav_content['params']['lidar_pose']
                t_list.append(transformation_utils.x_to_world(lidar_pose))  # Twx

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i != j:
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix

    def collate_batch_train(self, batch):
        # 'batch' is a list with a length of batch_size

        record_len = [] # used to record different scenario
        lidar_pose_list = []
        lidar_pose_clean_list = []
        pairwise_t_matrix_list = [] # pairwise transformation matrix
        processed_lidar_list = []
        image_inputs_list = []

        image_timestamps_list = []
        lidar_timestamps_list = []

        label_dict_list = []
        label_dict_list_single_v = []
        label_dict_list_single_i = []

        object_ids = []
        object_bbx_center = []
        object_bbx_mask = []

        object_ids_single_v = []
        object_bbx_center_single_v = []
        object_bbx_mask_single_v = []
        object_ids_single_i = []
        object_bbx_center_single_i = []
        object_bbx_mask_single_i = []

        if self.supervise_single:
            pos_equal_one_single = []
            neg_equal_one_single = []
            targets_single = []
        
        if self.kd_flag:
            teacher_processed_lidar_list = []
        
        if self.visualize:
            origin_lidar = []
            origin_lidar_v = []
            origin_lidar_i = []


        # emsemble batch data
        for i in range(len(batch)):
            ego_dict = batch[i]['ego']

            record_len.append(ego_dict['cav_num'])
            lidar_pose_list.append(ego_dict['lidar_poses'])     # shape = np.ndarray [N,6-DOF]
            lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.

            lidar_timestamps_list.append(ego_dict['lidar_timestamps'])
            image_timestamps_list.append(ego_dict['image_timestamps'])

            label_dict_list.append(ego_dict['label_dict'])
            label_dict_list_single_v.append(ego_dict['label_dict_single_v'])
            label_dict_list_single_i.append(ego_dict['label_dict_single_i'])

            object_ids.append(ego_dict['object_ids'])
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])

            object_ids_single_v.append(ego_dict['object_ids_single_v'])
            object_bbx_center_single_v.append(ego_dict['object_bbx_center_single_v'])
            object_bbx_mask_single_v.append(ego_dict['object_bbx_mask_single_v'])
            object_ids_single_i.append(ego_dict['object_ids_single_i'])
            object_bbx_center_single_i.append(ego_dict['object_bbx_center_single_i'])
            object_bbx_mask_single_i.append(ego_dict['object_bbx_mask_single_i'])
            
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                targets_single.append(ego_dict['single_label_dict_torch']['targets'])

            if self.kd_flag:
                teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
                origin_lidar_v.append(ego_dict['origin_lidar_v'])
                origin_lidar_i.append(ego_dict['origin_lidar_i'])
        
        # convert to numpy, (batch_size, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        object_bbx_center_single_v = torch.from_numpy(np.array(object_bbx_center_single_v))
        object_bbx_mask_single_v = torch.from_numpy(np.array(object_bbx_mask_single_v))

        object_bbx_center_single_i = torch.from_numpy(np.array(object_bbx_center_single_i))
        object_bbx_mask_single_i = torch.from_numpy(np.array(object_bbx_mask_single_i))

        # merge preprocessed features from different cavs into the same dict
        # merged_feature_dict = {'voxel_features': , 
        #                        'voxel_coords': , 
        #                        'voxel_num_points': }
        # merge and emsemble to union, and transform to torch
        # processed_lidar_torch_dict ={'voxel_features': shape=(sum(batch size*2), points per voxel, (x,y,z,intensity)), ...
        merged_feature_dict = merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
        # {"image_inputs": 
        #   {image: [sum(record_len), Ncam, C, H, W]}
        # }
        merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

        # emsemble timestamps
        lidar_timestamps_list = torch.from_numpy(np.array(lidar_timestamps_list))
        image_timestamps_list = torch.from_numpy(np.array(image_timestamps_list))

        # the sum of cav number per batch
        record_len = torch.from_numpy(np.array(record_len, dtype=int))

        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))

        label_torch_dict = self.post_processor.collate_batch(label_dict_list)
        label_torch_dict_single_v = self.post_processor.collate_batch(label_dict_list_single_v)
        label_torch_dict_single_i = self.post_processor.collate_batch(label_dict_list_single_i)

        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
        # add pairwise_t_matrix to label dict
        label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len
        label_torch_dict_single_v['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict_single_v['record_len'] = record_len
        label_torch_dict_single_i['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict_single_i['record_len'] = record_len
        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict = {
            'ego': {
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'object_ids': object_ids[0],
                'label_dict': label_torch_dict,
                'object_bbx_center_single_v': object_bbx_center_single_v,
                'object_bbx_mask_single_v': object_bbx_mask_single_v,
                'object_ids_single_v': object_ids_single_v[0],
                'label_dict_single_v': label_torch_dict_single_v,
                'object_bbx_center_single_i': object_bbx_center_single_i,
                'object_bbx_mask_single_i': object_bbx_mask_single_i,
                'object_ids_single_i': object_ids_single_i[0],
                'label_dict_single_i': label_torch_dict_single_i,

                'processed_lidar': processed_lidar_torch_dict,
                'image_inputs': merged_image_inputs_dict,
                'lidar_timestamp': lidar_timestamps_list,
                'image_timestamp': image_timestamps_list,
                'record_len': record_len,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_pose_clean': lidar_pose_clean,
                'lidar_pose': lidar_pose
            }
        }

        if self.kd_flag:
            teacher_processed_lidar_torch_dict = self.pre_processor.collate_batch(teacher_processed_lidar_list)
            output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})
        
        if self.visualize:
            origin_lidar = np.array(pcd_utils.downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

            origin_lidar_v = np.array(pcd_utils.downsample_lidar_minimum(pcd_np_list=origin_lidar_v))
            origin_lidar_v = torch.from_numpy(origin_lidar_v)
            output_dict['ego'].update({'origin_lidar_v': origin_lidar_v})
        
            origin_lidar_i = np.array(pcd_utils.downsample_lidar_minimum(pcd_np_list=origin_lidar_i))
            origin_lidar_i = torch.from_numpy(origin_lidar_i)
            output_dict['ego'].update({'origin_lidar_i': origin_lidar_i})

        ### 2022.10.10 single gt ####
        if self.supervise_single:
            output_dict['ego'].update({
                "label_dict_single" : 
                    {"pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                     "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                     "targets": torch.cat(targets_single, dim=0)}
            })

        output_dict['ego'].update({
            "mode": 'train'
        })

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box': torch.from_numpy(np.array(batch[0]['ego']['anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle transformation is only used in post process (no use.)
        # we all predict boxes in ego coord.
        output_dict['ego'].update({
            'transformation_matrix': torch.from_numpy(np.identity(4)).float(),
            'transformation_matrix_clean': torch.from_numpy(np.identity(4)).float()
        })

        output_dict['ego'].update({
            "sample_idx": batch[0]['ego']['sample_idx'],
            "cav_id_list": batch[0]['ego']['cav_id_list']
        })

        output_dict['ego'].update({
            "mode": 'test'
        })
        
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
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor
