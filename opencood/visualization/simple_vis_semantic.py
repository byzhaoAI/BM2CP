# -*- coding: utf-8 -*-
"""
CARLA Semantic
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import cv2
import numpy as np
import matplotlib.pyplot as plt


classes = {
        0: [255, 255, 255],  # None
        # 0: [0, 0, 0],  # None
        20: [70, 70, 70],  # Buildings
        2: [190, 153, 153],  # Fences
        3: [72, 0, 90],  # Other
        4: [220, 20, 60],  # Pedestrians
        5: [153, 153, 153],  # Poles
        6: [157, 234, 50],  # RoadLines
        7: [128, 64, 128],  # Roads
        8: [244, 35, 232],  # Sidewalks
        9: [107, 142, 35],  # Vegetation
        10: [0, 0, 255],  # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0],  # TrafficSigns
        13: [70, 130, 180],  # Sky
        14: [81, 0, 81],  # Ground
        15: [150, 100, 100],  # Bridge
        16: [230, 150, 140],  # RailTrack
        17: [180, 165, 180],  # All types of guard rails/crash barriers.
        18: [250, 170, 30],  # Traffic Light
        19: [110, 190, 160],  # Static
        1: [170, 120, 50],  # Dynamic
        21: [45, 60, 150],  # Water
        22: [145, 170, 100]  # Terrain
    }


def labels_to_palette(label):
    """
    Convert an image containing semantic segmentation labels to palette.

    Parameters
    ----------
    label : np.ndarray label image, (h, w)

    Returns
    -------
    Converted BGR image.
    """
    result = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for key, value in classes.items():
        result[np.where(label == key)] = value
    
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

def visualize(pred, label, save_path, gt_save_path=None):
    img = labels_to_palette(pred)
    gt_img = labels_to_palette(label)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, img)
    
    if gt_save_path is not None:
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(gt_save_path, gt_img)
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.xticks([])  # 移除 x 轴刻度值
    # plt.yticks([])  # 移除 y 轴刻度值

    # plt.subplot(1, 2, 2)
    # plt.imshow(gt_img)
    # plt.xticks([])  # 移除 x 轴刻度值
    # plt.yticks([])  # 移除 y 轴刻度值

    # plt.axis("off")
    # plt.tight_layout()

    # plt.savefig(save_path, dpi=400)
    # plt.savefig(save_path, transparent=False, dpi=400)
    # plt.close()


def visualize_road(pred, label, save_path):
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(gt_img, cmap='gray')

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, transparent=True, dpi=400)
    plt.close()
