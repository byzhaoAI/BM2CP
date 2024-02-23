# -*- coding: utf-8 -*-
# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>
# Author: Quanhao Li <quanhaoli2022@163.com> Yifan Lu <yifan_lu@sjtu.edu.cn>, 
# Author: Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for intermediate fusion (DAIR-V2X)
"""
import math
import copy
import os
import numpy as np
import json
import torch
from collections import OrderedDict

from opencood.data_utils import pre_processor, post_processor, augmentor
from opencood.utils import pcd_utils, box_utils, pose_utils, transformation_utils


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def merge_features_to_dict(processed_feature_list):
    """
    Merge the preprocessed features from different cavs to the same
    dictionary.

    Parameters
    ----------
    processed_feature_list : list
        A list of dictionary containing all processed features from
        different cavs.

    Returns
    -------
    merged_feature_dict: dict
        key: feature names, value: list of features.
    """

    merged_feature_dict = OrderedDict()

    for i in range(len(processed_feature_list)):
        for feature_name, feature in processed_feature_list[i].items():
            # print(i, feature_name, type(feature))

            if feature_name not in merged_feature_dict:
                merged_feature_dict[feature_name] = []
            
            # merged_feature_dict['coords'] = [f1,f2,f3,f4]
            if isinstance(feature, list):
                merged_feature_dict[feature_name] += feature
            else:
                merged_feature_dict[feature_name].append(feature)

    return merged_feature_dict


class IntermediateFusionDatasetDAIR(torch.utils.data.Dataset):
    """
    This class is for intermediate fusion where each vehicle transmit the deep features to ego.
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

        # pre- and post- precessor, data augmentor
        self.pre_processor = pre_processor.build_preprocessor(params['preprocess'], train)
        self.post_processor = post_processor.build_postprocessor(params['postprocess'], dataset='dair', train=train)
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
        coop_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        for frame_info in coop_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.coop_data[veh_frame_id] = frame_info

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        # base_data_dict = {0: {'ego':{True}, 'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_np':}},  # for vehicle
        #                   1: {'ego':{False}, 'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_np':}}}  # for infrastructure
        base_data_dict = self.retrieve_base_data(idx)

        # base_data_dict = {0: {'ego':{True}, 'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_np':, 'lidar_pose_clean':,}},  # for vehicle
        #                   1: {'ego':{False}, 'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_np':, 'lidar_pose_clean':,}}}  # for infrastructure
        base_data_dict = pose_utils.add_noise_data_dict(base_data_dict, self.params['noise_setting'])
        assert len(base_data_dict[0]['params']['lidar_pose']) > 0

        # data prepared for each batch
        cav_id_list = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        processed_features = []
        projected_lidar_clean_list = []

        object_id_stack = []
        object_id_stack_single_v = []
        object_id_stack_single_i = []
        object_stack = []
        object_stack_single_v = []
        object_stack_single_i = []
        
        if self.visualize:
            projected_lidar_stack = []
        

        # filter the agents that out of communicate range, sqrt(x**2+y**2)
        filtered_cav_id = []
        for cav_id, selected_cav_base in base_data_dict.items():
            # distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + 
            #                      (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            # if distance > self.params['comm_range']:
            #     filtered_cav_id.append(cav_id)
            #     continue

            cav_id_list.append(cav_id)
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
        
        for cav_id in filtered_cav_id:
            base_data_dict.pop(cav_id)


        # loop over all CAVs to process information
        for cav_id in cav_id_list:
            selected_cav_processed = self.get_item_single_car(base_data_dict[cav_id], base_data_dict[0]['params']['lidar_pose'], base_data_dict[0]['params']['lidar_pose_clean'])
            
            # selected_cav_processed['object_bbx_center']: box for cooperative
            # selected_cav_processed['object_bbx_center_single']: box for vehicle or infrastructure, is not equal to cooperative
            object_id_stack += selected_cav_processed['object_ids']
            object_stack.append(selected_cav_processed['object_bbx_center'])
            if cav_id == 0: # ego vehicle load cooperative label (id&bbx)
                object_id_stack_single_v += selected_cav_processed['object_ids_single']
                object_stack_single_v.append(selected_cav_processed['object_bbx_center_single'])
            else:           # infrastructure vehicle load local label (id&bbx)
                object_id_stack_single_i += selected_cav_processed['object_ids_single']
                object_stack_single_i.append(selected_cav_processed['object_bbx_center_single'])

            processed_features.append(selected_cav_processed['processed_features'])

            if self.kd_flag:
                projected_lidar_clean_list.append(selected_cav_processed['projected_lidar_clean'])

            if self.visualize:
                projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
        
        # merge preprocessed features from different cavs into the same dict
        # merged_feature_dict = {'voxel_features': shape=(voxel numbers, max points per voxel, (x,y,z,intensity)), 
        #                        'voxel_coords': , 
        #                        'voxel_num_points': }
        merged_feature_dict = merge_features_to_dict(processed_features)
        
        if self.kd_flag:    # for disconet knowledge distillation
            stack_lidar_np = np.vstack(projected_lidar_clean_list)
            stack_lidar_np = pcd_utils.mask_points_by_range(stack_lidar_np, self.params['preprocess']['cav_lidar_range'])
            stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)

        # generate the anchor boxes and target label
        # object_stack: length=2*batch_size (1/2 valid), object_stack_single_v: length=batch_size, object_stack_single_i: length=batch_size
        # label_dict = {'pos_equal_one': shape=(100, 252, 2),
        #               'neg_equal_one': shape=(100, 252, 2),   # unused label item
        #               'targets': shape=(100, 252, 14)
        #               }
        object_bbx_center, mask, object_id_stack = self.get_unique_label(object_stack, object_id_stack)
        object_bbx_center_single_v, mask_single_v, object_id_stack_single_v = self.get_unique_label(object_stack_single_v, object_id_stack_single_v)
        object_bbx_center_single_i, mask_single_i, object_id_stack_single_i = self.get_unique_label(object_stack_single_i, object_id_stack_single_i)

        anchor_box = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask)
        label_dict_single_v = self.post_processor.generate_label(gt_box_center=object_bbx_center_single_v, anchors=anchor_box, mask=mask_single_v)
        label_dict_single_i = self.post_processor.generate_label(gt_box_center=object_bbx_center_single_i, anchors=anchor_box, mask=mask_single_i)
        

        # build dictionary for dataset output
        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {
            'sample_idx': idx, 
            'cav_num': len(cav_id_list),
            'cav_id_list': cav_id_list,
            'pairwise_t_matrix': self.get_pairwise_transformation(base_data_dict, self.max_cav),    # get pairwise transformation, matrix[i, j] is i to j
            'lidar_poses': np.array(lidar_pose_list).reshape(-1, 6),  # [N_cav, 6]
            'lidar_poses_clean': np.array(lidar_pose_clean_list).reshape(-1, 6),  # [N_cav, 6],
            'processed_lidar': merged_feature_dict,

            'label_dict': label_dict,
            'label_dict_single_v': label_dict_single_v,
            'label_dict_single_i': label_dict_single_i,

            'anchor_box': anchor_box,
            'object_ids': object_id_stack,
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': mask,
            'object_ids_single_v': object_id_stack_single_v,
            'object_bbx_center_single_v': object_bbx_center_single_v,
            'object_bbx_mask_single_v': mask_single_v,
            'object_ids_single_i': object_id_stack_single_i,
            'object_bbx_center_single_i': object_bbx_center_single_i,
            'object_bbx_mask_single_i': mask_single_i,
        }

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
        """
        # veh_frame_id: frame id of vehicle in train.json. e.g. "000010", "000011", "000013"
        # frame_info: img&point cloud of infrastructure&vehicle in vehicle info. 
        #       e.g. {"infrastructure_image_path": ..., 
        #             "infrastructure_pointcloud_path": ...,
        #             ...,
        #             "system_error_offset": {"delta_x": ..., "delta_y": ...}}
        veh_frame_id = self.split_info[idx]
        frame_info = self.coop_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]

        # build dictionary for each agent
        #       e.g. {0: {'ego':{True}, 'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_np':}},  # for vehicle
        #             1: {'ego':{False}, 'params': {'vehicles':, 'lidar_pose':, 'vehicles_single':, 'lidar_np':}}  # for infrastructure
        #            }
        data = OrderedDict()
        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[0]['params'] = OrderedDict()
        data[1] = OrderedDict()
        data[1]['ego'] = False
        data[1]['params'] = OrderedDict()


        # vehicle-side
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir, frame_info['cooperative_label_path']))
        
        # 6-DOF pose
        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        transformation_matrix = transformation_utils.veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)
        data[0]['params']['lidar_pose'] = transformation_utils.tfm_to_pose(transformation_matrix)
        
        # label in single view
        vehicle_side_path = os.path.join(self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(veh_frame_id))
        data[0]['params']['vehicles_single'] = load_json(vehicle_side_path) 

        # get (point cloud numpy & time)
        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
        if self.clip_pc:
            data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:,0] > 0]
        

        # infrastructure-side
        # only load cooperative once in vehicle side
        data[1]['params']['vehicles'] = []

        # 6-DOF pose
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix1 = transformation_utils.inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file, system_error_offset)
        data[1]['params']['lidar_pose'] = transformation_utils.tfm_to_pose(transformation_matrix1)   

        # label in single view
        infra_side_path = os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id))
        data[1]['params']['vehicles_single'] = load_json(infra_side_path)

        # get (point cloud numpy & time)
        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))

        return data

    def get_item_single_car(self, selected_cav_base, ego_pose, ego_pose_clean):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        ego_pose_clean : list, length 6
            only used for gt box generation

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
            {
                'object_bbx_center': ,              # the exact number of object
                'object_ids': ,
                'object_bbx_center_single': ,       # the exact number of object
                'object_ids_single': ,
                'transformation_matrix': ,
                'transformation_matrix_clean': , 
                'processed_features': ,             # voxel feature data
                'projected_lidar': ,                # raw point cloud data
                'projected_lidar_clean':            # (optional) raw point cloud data
             }
        """
        # calculate the transformation matrix, the transformation matrix from x1(other) to x2(ego)
        transformation_matrix = transformation_utils.x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose)
        transformation_matrix_clean = transformation_utils.x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)

        # retrieve objects under ego coordinates. this is used to generate accurate GT bounding box.
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center_dairv2x(selected_cav_base, ego_pose_clean)
        object_bbx_center_single, object_bbx_mask_single, object_ids_single = self.post_processor.generate_object_center_dairv2x_late_fusion(selected_cav_base)
        
        # filter lidar. remove the lidar points of the ego vehicle itself.
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = pcd_utils.shuffle_points(lidar_np)
        lidar_np = pcd_utils.mask_ego_points(lidar_np)
        # project the lidar to ego space. only x,y,z
        # this function transforms cloud coordinate (x,y,z) to ego coordinate (x,y,z), preserve intensity information always
        projected_lidar = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)

        if self.proj_first:
            lidar_np[:, :3] = projected_lidar

        # filter points out of requirement range
        lidar_np = pcd_utils.mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])

        # point cloud -> voxel feature
        # processed_features: shape=(voxel numbers, max points per voxel, (x,y,z,intensity))
        processed_features = self.pre_processor.preprocess(lidar_np)

        selected_cav_processed = {
            'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
            'object_ids': object_ids,
            'object_bbx_center_single': object_bbx_center_single[object_bbx_mask_single == 1],
            'object_ids_single': object_ids_single,
            'transformation_matrix': transformation_matrix,
            'transformation_matrix_clean': transformation_matrix_clean,
            'processed_features': processed_features,
            'projected_lidar': projected_lidar,
        }

        if self.kd_flag:
            lidar_np_clean = pcd_utils.mask_ego_points(selected_cav_base['lidar_np']) # copy.deepcopy(lidar_np)
            projected_lidar_clean = box_utils.project_points_by_matrix_torch(lidar_np_clean[:, :3], transformation_matrix_clean)
            lidar_np_clean[:, :3] = projected_lidar_clean
            lidar_np_clean = pcd_utils.mask_points_by_range(lidar_np_clean, self.params['preprocess']['cav_lidar_range'])
            selected_cav_processed.update({"projected_lidar_clean": lidar_np_clean})    # point cloud data

        return selected_cav_processed

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
            processed_lidar_list.append(ego_dict['processed_lidar']) # different cav_num, ego_dict['processed_lidar'] is list.

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

        # object id is only used during inference, where batch size is 1. so here we only get the first element of list.
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

        if self.params['preprocess']['core_method'] == 'SpVoxelPreprocessor' and (output_dict['ego']['processed_lidar']['voxel_coords'][:, 0].max().int().item() + 1) != record_len.sum().int().item():
            return None

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
