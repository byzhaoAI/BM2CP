# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
# 设置环境变量以同步执行 CUDA 内核
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import statistics

import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)

import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from thop import profile

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--epoch', type=int)
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    print('Creating Model')
    model = train_utils.create_model(hypes)

    saved_path = opt.model_dir
    init_epoch, model = train_utils.load_saved_model(saved_path, model, opt.epoch)

    # 查看模型中的所有可训练参数
    print("\nParameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param}")

if __name__ == '__main__':
    main()
