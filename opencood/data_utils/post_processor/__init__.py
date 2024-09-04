# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.data_utils.post_processor.voxel_postprocessor2 import VoxelPostprocessor2

__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor,
    'VoxelPostprocessor2': VoxelPostprocessor2
}


def build_postprocessor(anchor_cfg, dataset, train):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in ['VoxelPostprocessor', 'VoxelPostprocessor2']
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg,
        dataset=dataset,
        train=train,
    )

    return anchor_generator