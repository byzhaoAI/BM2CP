"""
Dataset class for early fusion
"""
import os
import math
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

import opencood
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points, downsample_lidar_minimum

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.datasets.opv2v import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils, pcd_utils, transformation_utils, camera_utils, sensor_transformation_utils



def merge_features_to_dict(processed_feature_list, merge=None):
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

            if feature_name not in merged_feature_dict:
                merged_feature_dict[feature_name] = []
            if isinstance(feature, list):
                merged_feature_dict[feature_name] += feature
            else:
                merged_feature_dict[feature_name].append(feature)

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


class LiDARCameraIntermediateFusionDataset(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True):
        super(LiDARCameraIntermediateFusionDataset, self).__init__(params, visualize, train)
        self.params = params
        self.visualize = visualize
        self.train = train

        self.data_augmentor = DataAugmentor(params['data_augment'], train)
        self.pre_processor = build_preprocessor(params['preprocess'], train)
        self.post_processor = post_processor.build_postprocessor(params['postprocess'], dataset='opv2v', train=train)

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']

        self.frame = params['train_params']['frame']
        
        # if project first, cav's lidar will first be projected to the ego's coordinate frame. otherwise, the feature will be projected instead.
        self.proj_first = params['fusion']['args']['proj_first'] if 'proj_first' in params['fusion']['args'] else False
        print('proj_first: ', self.proj_first)
        # whether there is a time delay between the time that cav project lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = True if 'cur_ego_pose_flag' not in params['fusion']['args'] else params['fusion']['args']['cur_ego_pose_flag']
        
        self.grid_conf = params["fusion"]["args"]["grid_conf"]
        self.depth_discre = camera_utils.depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode'])
        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

    def __getitem__(self, idx):
        select_num = self.frame
        select_dict,scenario_index,index_list,timestamp_index = self.retrieve_multi_data(idx,select_num,cur_ego_pose_flag=self.cur_ego_pose_flag)
        if timestamp_index < select_num:
            idx += select_num 
        try:
            assert idx == list(select_dict.keys())[0], "The first element in the multi frame must be current index"
        except AssertionError as aeeor:
            print("assert error dataset",list(select_dict.keys()),idx,timestamp_index)
        
        processed_data_list = []
        ego_id = -1
        ego_lidar_pose = []
        ego_id_list = []
        
        cav_num_list = []
        cav_list = []
        for index, base_data_dict in select_dict.items():
            if index == idx:
                # first find the ego vehicle's lidar pose
                for cav_id, cav_content in base_data_dict.items():
                    if cav_content['ego']:
                        ego_id = cav_id
                        ego_lidar_pose = cav_content['params']['lidar_pose']
                        break
                assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0
            ego_id_list.append(ego_id)
            # this is used for v2vnet and disconet
            pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.params['train_params']['max_cav'])

            cav_id_list = []
            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + 
                                     (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
                if distance > opencood.data_utils.datasets.COM_RANGE:
                    continue
                cav_id_list.append(cav_id)
            cav_num_list.append(len(cav_id_list))
            cav_list.append(cav_id_list)

        try:
            assert len(set(ego_id_list)) == 1, "The ego id must be same"
        except AssertionError as aeeor:
            print("assert error ego id",ego_id_list)

        min_cav_num = min(cav_num_list)
        min_cav_index = cav_num_list.index(min_cav_num)
        cav_ids = cav_list[min_cav_index]
        # if the number of agents is not same in different frame
        if np.count_nonzero(np.array(cav_num_list) - np.ones(len(cav_num_list)) * min_cav_num) > 0:
            for _id in cav_ids:
                for cav_id in cav_list:
                    assert _id in cav_id, f"{id} should be in {cav_id}"

        for index, base_data_dict in select_dict.items():
            agents_image_inputs = []
            processed_features = []
            object_stack = []
            object_id_stack = []

            # prior knowledge for time delay correction and indicating data type (V2V vs V2i)
            velocity = []
            time_delay = []
            infra = []
            spatial_correction_matrix = []

            if self.visualize:
                projected_lidar_stack = []

            lidar_pose = []
            ego_flag = []

            # loop over all CAVs to process information
            """
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + 
                                     (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
                if distance > opencood.data_utils.datasets.COM_RANGE:
                    continue
            """
            for _idx, cav_id in enumerate(cav_ids):
                selected_cav_base = base_data_dict[cav_id]


                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)

                object_id_stack += selected_cav_processed['object_ids']
                object_stack.append(selected_cav_processed['object_bbx_center'])
                processed_features.append(selected_cav_processed['processed_features'])
                agents_image_inputs.append(selected_cav_processed['image_inputs'])
                velocity.append(selected_cav_processed['velocity'])
                time_delay.append(float(selected_cav_base['time_delay']))
                spatial_correction_matrix.append(selected_cav_base['params']['spatial_correction_matrix'])
                infra.append(1 if int(cav_id) < 0 else 0)
                cav_id_list.append(cav_id)

                lidar_pose.append(selected_cav_base['params']['lidar_pose'])
                if ego_id == cav_id:
                    ego_flag.append(True)
                else:
                    ego_flag.append(False)
                if self.visualize:
                    projected_lidar_stack.append(selected_cav_processed['projected_lidar'])

                if _idx == min_cav_num - 1:
                    break

            # merge preprocessed features from different cavs into the same dict
            merged_feature_dict = merge_features_to_dict(processed_features)
            merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
            
            # exclude all repetitive objects
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask)

            # pad dv, dt, infra to max_cav
            velocity = velocity + (self.max_cav - len(velocity)) * [0.]
            time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
            infra = infra + (self.max_cav - len(infra)) * [0.]
            spatial_correction_matrix = np.stack(spatial_correction_matrix)
            padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(spatial_correction_matrix),1,1))
            spatial_correction_matrix = np.concatenate([spatial_correction_matrix, padding_eye], axis=0)

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {
                'cav_num': len(processed_features),
                'lidar_pose': lidar_pose,
                'pairwise_t_matrix': pairwise_t_matrix,
                'spatial_correction_matrix': spatial_correction_matrix,
                'image_inputs': merged_image_inputs_dict,
                'processed_lidar': merged_feature_dict,
                'label_dict': label_dict,
                'anchor_box': anchor_box,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'velocity': velocity,
                'time_delay': time_delay,
                'infra': infra,
                'ego_flag': ego_flag
            }

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar': np.vstack(projected_lidar_stack)})
            processed_data_list.append(processed_data_dict)

        return processed_data_list

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
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
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
            timestamp_key_delay = self.return_timestamp_key(scenario_database, timestamp_index_delay)

            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag)
            data[cav_id]['lidar_np'] = pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
            img_src = []
            for idx in range(self.data_aug_conf['Ncams']):
                image_path = cav_content[timestamp_key_delay]['camera0'][idx]
                img_src.append(Image.open(image_path))
            data[cav_id]['camera_data'] = img_src
        return data, scenario_index, timestamp_key

    def retrieve_base_data_before(self, scenario_index, idx, cur_timestamp_key, cur_ego_pose_flag=True):
        scenario_database = self.scenario_database[scenario_index]  

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)  

        data = OrderedDict()
        # load files for all CAVs self.scenario_database[i][cav_id]['ego'] = True
        for cav_id, cav_content in scenario_database.items(): 
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            timestamp_delay = self.time_delay_calculation(cav_content['ego'])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database, timestamp_index_delay) 
            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary 
            data[cav_id]['params'] = self.reform_param(cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag)
            data[cav_id]['lidar_np'] = pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
            img_src = []
            for idx in range(self.data_aug_conf['Ncams']):
                image_path = cav_content[timestamp_key_delay]['camera0'][idx]
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

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

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
        # for key, value in selected_cav_processed['image_inputs'].items():
        #     print(value.shape)

        # process lidar data

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base['params']['transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center([selected_cav_base], ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        
        lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})

        return selected_cav_processed


    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict_list = []
        for j in range(len(batch[0])):

            record_len = [] # used to record different scenario
            lidar_poses = []

            pairwise_t_matrix_list = [] # pairwise transformation matrix
            processed_lidar_list = []
            image_inputs_list = []
            label_dict_list = []

            object_ids = []
            object_bbx_center = []
            object_bbx_mask = []

            # used for PriorEncoding
            velocity = []
            time_delay = []
            infra = []

            # used for correcting the spatial transformation between delayed timestamp
            # and current timestamp
            spatial_correction_matrix_list = []
            ego_flag = []

            if self.visualize:
                origin_lidar = []

            for i in range(len(batch)):
                ego_dict = batch[i][j]['ego']
                record_len.append(ego_dict['cav_num'])
                lidar_poses.append(ego_dict['lidar_pose'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
                processed_lidar_list.append(ego_dict['processed_lidar'])
                image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                label_dict_list.append(ego_dict['label_dict'])

                object_ids.append(ego_dict['object_ids'])
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask']) 

                velocity.append(ego_dict['velocity'])
                time_delay.append(ego_dict['time_delay'])
                infra.append(ego_dict['infra'])
                spatial_correction_matrix_list.append(ego_dict['spatial_correction_matrix'])

                ego_flag.append(ego_dict['ego_flag'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

            lidar_poses = torch.from_numpy(np.concatenate(lidar_poses, axis=0))

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            # example: {'voxel_features':[np.array([1,2,3]]),
            # np.array([3,5,6]), ...]}
            merged_feature_dict = merge_features_to_dict(processed_lidar_list)
            processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
            merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')
            # [2, 3, 4, ..., M]
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            label_torch_dict = self.post_processor.collate_batch(label_dict_list)

            # (B, max_cav)
            velocity = torch.from_numpy(np.array(velocity))
            time_delay = torch.from_numpy(np.array(time_delay))
            infra = torch.from_numpy(np.array(infra))
            spatial_correction_matrix_list = torch.from_numpy(np.array(spatial_correction_matrix_list))
            # (B, max_cav, 3)
            prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict = {
                'ego': {
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask,
                    'object_ids': object_ids[0],
                    'label_dict': label_torch_dict,
                    'processed_lidar': processed_lidar_torch_dict,
                    'image_inputs': merged_image_inputs_dict,
                    'record_len': record_len,
                    'prior_encoding': prior_encoding,
                    'spatial_correction_matrix': spatial_correction_matrix_list,
                    'pairwise_t_matrix': pairwise_t_matrix,
                    'lidar_pose': lidar_poses,
                    'ego_flag': ego_flag
                }
            }

            if self.visualize:
                origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})
            output_dict_list.append(output_dict)

        return output_dict_list

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict_list = self.collate_batch_train(batch)

        # check if anchor box in the batch
        for i in range(len(batch[0])):
            if batch[0][i]['ego']['anchor_box'] is not None:
                output_dict_list[i]['ego'].update({'anchor_box': torch.from_numpy(np.array(batch[0][i]['ego']['anchor_box']))})

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
            output_dict_list[i]['ego'].update({'transformation_matrix': transformation_matrix_torch})

        return output_dict_list

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

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix across different agents.
        This is only used for v2vnet and disconet. Currently we set
        this as identity matrix as the pointcloud is projected to
        ego vehicle first.

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
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        
        if self.proj_first:
            # default are identity matrix
            pairwise_t_matrix[:, :] = np.identity(4)

            return pairwise_t_matrix

        # save all transformation matrix in a list in order first.
        t_list = []
        for cav_id, cav_content in base_data_dict.items():
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    t_matrix = np.eye(4)
                    pairwise_t_matrix[i, j] = t_matrix
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
