def retrieve_base_data_before(self, scenario_index, idx, cur_timestamp_key, cur_ego_pose_flag=True):
        
        scenario_database = self.scenario_database[scenario_index]  

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = \
            self.calc_dist_to_ego(scenario_database, timestamp_key)  

        data = OrderedDict()
        # load files for all CAVs self.scenario_database[i][cav_id]['ego'] = True
        for cav_id, cav_content in scenario_database.items(): 
            if cav_content['ego']: 
                data[cav_id] = OrderedDict()
                data[cav_id]['ego'] = cav_content['ego']

                # calculate delay for this vehicle
                timestamp_delay = \
                    self.time_delay_calculation(cav_content['ego'])

                if timestamp_index - timestamp_delay <= 0:
                    timestamp_delay = timestamp_index
                timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
                timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                                timestamp_index_delay) 
                # add time delay to vehicle parameters
                data[cav_id]['time_delay'] = timestamp_delay
                # load the corresponding data into the dictionary 
                data[cav_id]['params'] = self.reform_param(cav_content,
                                                        ego_cav_content,
                                                        cur_timestamp_key,
                                                        timestamp_key_delay,
                                                        cur_ego_pose_flag)
                data[cav_id]['lidar_np'] = \
                    pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
        return data