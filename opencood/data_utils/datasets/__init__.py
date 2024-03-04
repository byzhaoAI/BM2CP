# -*- coding: utf-8 -*-
# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.opv2v.early_fusion_dataset import EarlyFusionDataset as EarlyFusionDatasetOPV2V
from opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset import IntermediateFusionDataset as IntermediateFusionDatasetOPV2V
from opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2 as IntermediateFusionDatasetOPV2V_V2
from opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset_multi_frame import IntermediateFusionDataset as IntermediateFusionDatasetOPV2V_MULTI
from opencood.data_utils.datasets.opv2v.late_fusion_dataset import LateFusionDataset as LateFusionDatasetOPV2V
from opencood.data_utils.datasets.opv2v.lidar_camera_intermediate_fusion_dataset import LiDARCameraIntermediateFusionDataset as LiDARCameraIntermediateFusionDatasetOPV2V
from opencood.data_utils.datasets.opv2v.lidar_camera_intermediate_fusion_dataset_v2 import LiDARCameraIntermediateFusionDataset as LiDARCameraIntermediateFusionDatasetOPV2V_V2

from opencood.data_utils.datasets.dair.early_fusion_dataset import EarlyFusionDatasetDAIR
from opencood.data_utils.datasets.dair.intermediate_fusion_dataset import IntermediateFusionDatasetDAIR
from opencood.data_utils.datasets.dair.intermediate_fusion_dataset_multi_frame import IntermediateFusionDatasetDAIR as IntermediateFusionDatasetDAIR_MULTI
from opencood.data_utils.datasets.dair.late_fusion_dataset import LateFusionDatasetDAIR
from opencood.data_utils.datasets.dair.lidar_camera_intermediate_fusion_dataset import LiDARCameraIntermediateFusionDatasetDAIR
from opencood.data_utils.datasets.dair.lidar_camera_intermediate_fusion_dataset_v2 import LiDARCameraIntermediateFusionDatasetDAIR as LiDARCameraIntermediateFusionDatasetDAIR_V2

from opencood.data_utils.datasets.v2v4real.early_fusion_dataset import EarlyFusionDataset as EarlyFusionDatasetV2V4Real
from opencood.data_utils.datasets.v2v4real.late_fusion_dataset import LateFusionDataset as LateFusionDatasetV2V4Real
from opencood.data_utils.datasets.v2v4real.intermediate_fusion_dataset import IntermediateFusionDataset as IntermediateFusionDatasetV2V4Real

__all__ = {
    'EarlyFusionDatasetOPV2V': EarlyFusionDatasetOPV2V,
    'IntermediateFusionDatasetOPV2V': IntermediateFusionDatasetOPV2V,
    'IntermediateFusionDatasetOPV2V_V2': IntermediateFusionDatasetOPV2V_V2,
    'IntermediateFusionDatasetOPV2V_Multi': IntermediateFusionDatasetOPV2V_MULTI,
    'LateFusionDatasetOPV2V': LateFusionDatasetOPV2V,
    'LiDARCameraIntermediateFusionDatasetOPV2V': LiDARCameraIntermediateFusionDatasetOPV2V,
    'LiDARCameraIntermediateFusionDatasetOPV2V_V2': LiDARCameraIntermediateFusionDatasetOPV2V_V2,

    'EarlyFusionDatasetDAIR': EarlyFusionDatasetDAIR,
    'IntermediateFusionDatasetDAIR': IntermediateFusionDatasetDAIR,
    'IntermediateFusionDatasetDAIR_Multi': IntermediateFusionDatasetDAIR_MULTI,
    'LateFusionDatasetDAIR': LateFusionDatasetDAIR,
    'LiDARCameraIntermediateFusionDatasetDAIR': LiDARCameraIntermediateFusionDatasetDAIR,
    'LiDARCameraIntermediateFusionDatasetDAIRV2': LiDARCameraIntermediateFusionDatasetDAIR_V2,

    'EarlyFusionDatasetV2V4Real': EarlyFusionDatasetV2V4Real,
    'IntermediateFusionDatasetV2V4Real': IntermediateFusionDatasetV2V4Real,
    'LateFusionDatasetV2V4Real': LateFusionDatasetV2V4Real,
}

# the final range for evaluation
GT_RANGE_OPV2V = [-140, -40, -3, 140, 40, 1]
GT_RANGE_V2XSIM = [-32, -32, -3, 32, 32, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
