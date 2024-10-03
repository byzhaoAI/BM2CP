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
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--random', default=False,
                        help='Random modality training')
    parser.add_argument('--from_init', default=False,
                        help='Continued training or init trianing')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    train_loader = DataLoader(opencood_train_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=16,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=16
                            )
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=16,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=16
                            )

    print('Creating Model')
    model = train_utils.create_model(hypes)
    f1_net, f2_net, f3_net = None, None, None
    shrink_conv, backbone, cls_head, reg_head, dir_head = None, None, None, None, None
    if 'f1' in hypes['model']['args'] and 'model_path' in hypes['model']['args']['f1']:
        f1_net = train_utils.create_model(hypes)
        _, f1_net = train_utils.load_saved_model(hypes['model']['args']['f1']['model_path'], f1_net)
        if hypes['model']['args']['f1']['freeze']:
            shrink_conv, backbone, cls_head, reg_head, dir_head = f1_net.shrink_conv, f1_net.pyramid_backbone, f1_net.cls_head, f1_net.reg_head, f1_net.dir_head
            f1_net = f1_net.f1
    if 'f2' in hypes['model']['args'] and 'model_path' in hypes['model']['args']['f2']:
        f2_net = train_utils.create_model(hypes)
        _, f2_net = train_utils.load_saved_model(hypes['model']['args']['f2']['model_path'], f2_net)
        if hypes['model']['args']['f2']['freeze']:
            shrink_conv, backbone, cls_head, reg_head, dir_head = f2_net.shrink_conv, f2_net.pyramid_backbone, f2_net.cls_head, f2_net.reg_head, f2_net.dir_head
            f2_net = f2_net.f2
    if 'f3' in hypes['model']['args'] and 'model_path' in hypes['model']['args']['f3']:
        f3_net = train_utils.create_model(hypes)
        _, f3_net = train_utils.load_saved_model(hypes['model']['args']['f3']['model_path'], f3_net)
        if hypes['model']['args']['f3']['freeze']:
            shrink_conv, backbone, cls_head, reg_head, dir_head = f3_net.shrink_conv, f3_net.pyramid_backbone, f3_net.cls_head, f3_net.reg_head, f3_net.dir_head
            f3_net = f3_net.f3
    model.update_model(shrink_conv, backbone, cls_head, reg_head, dir_head, f1_net, f2_net, f3_net)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %d" % (total))
    # print(model)
    # print("Number of parameter: %.2fM" % (total/1e6))
    
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)


    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        if opt.from_init:
            init_epoch = 0
            print('open init epoch from 0.')
        lowest_val_epoch = init_epoch ###
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch, n_iter_per_epoch=len(train_loader))
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        print("output result save to: ", saved_path)
        # lr scheduler setup
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, n_iter_per_epoch=len(train_loader))
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    with_round_loss = False
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * len(train_loader) + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        #shuffle for distributed, 在dataloader中加入打乱顺序（shuffle）的操作
        # if hypes['name'] == 'dair_v2xvit':
            # DistributedSampler(opencood_train_dataset).set_epoch(epoch) 
        for i, batch_data in enumerate(train_loader):
            collect_unit_loss = []
            mode = -1
            # print("batch_data: ", batch_data['ego'].keys())
            if batch_data is None:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            if 'scope' in hypes['name'] or 'how2comm' in hypes['name'] or 'syncnet' in hypes['name']:
                _batch_data = batch_data[0]
                batch_data = train_utils.to_device(batch_data, device)
                _batch_data = train_utils.to_device(_batch_data, device)

                ouput_dict = model(batch_data)
                final_loss = criterion(ouput_dict, _batch_data['ego']['label_dict'])
            else:
                batch_data = train_utils.to_device(batch_data, device)
                # case1 : late fusion train --> only ego needed,
                # and ego is (random) selected
                # case2 : early fusion train --> all data projected to ego
                # case3 : intermediate fusion --> ['ego']['processed_lidar']
                # becomes a list, which containing all data from other cavs
                # as well
                batch_data['ego']['epoch'] = epoch

                if 'vqm' in hypes['name']:
                    if opt.random:
                        mode = np.random.choice(2, 2)
                        output_dict = model(batch_data['ego'], mode, training=True)
                    else:
                        output_dict = model(batch_data['ego'], training=True)
                elif 'heal' in hypes['name']:
                    output_dict = model(batch_data['ego'], single_train=True)
                else:
                    output_dict = model(batch_data['ego'])
                #print(output_dict.keys())
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                collect_unit_loss = [output_dict['rec_loss'].item(), output_dict['svd_loss'].item(), output_dict['bfp_loss'].item()]
                # if 'modality_num' in output_dict and output_dict['modality_num'] > 1:
                #     for m_idx in range(output_dict['modality_num']):
                #         unit_loss = criterion(output_dict, batch_data['ego']['label_dict'], suffix='_{}'.format(m_idx))
                #         final_loss = final_loss + unit_loss
                #         collect_unit_loss.append(unit_loss.item())
                final_loss = criterion.rec_forward(output_dict, final_loss)
                if hypes['model']['args']['supervise_single']:
                    if 'dair' in opt.hypes_yaml:
                        single_loss_v = criterion(output_dict, batch_data['ego']['label_dict_single_v'], prefix='_single_v')
                        single_loss_i = criterion(output_dict, batch_data['ego']['label_dict_single_i'], prefix='_single_i')
                        final_loss = final_loss + single_loss_v + single_loss_i
                        collect_unit_loss = collect_unit_loss + [single_loss_v.item(), single_loss_i.item()]

            # criterion.logging(epoch, i, len(train_loader), writer)
            criterion.logging(epoch+1, i, len(train_loader), writer)    
            
            with open(os.path.join(saved_path, 'train_loss.txt'), 'a+') as f:
                msg = "[epoch %d][%d/%d][%s] || Total || %.5f, Rec: %.5f, SVD: %.5f, BFP: %.5f || " % (
                  epoch+1, i+1, len(train_loader), mode,
                  final_loss, collect_unit_loss[0], collect_unit_loss[1], collect_unit_loss[2])
                msg += f"Others: {collect_unit_loss[3:]} \n"
                f.write(msg)

            #print(a)
            # back-propagation
            final_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * len(train_loader) + i)
        
        if (epoch+1) % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if (epoch+1) % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    if 'scope' in hypes['name'] or 'how2comm' in hypes['name'] or 'fecp' in hypes['name']:
                        _batch_data = batch_data[0]
                        batch_data = train_utils.to_device(batch_data, device)
                        _batch_data = train_utils.to_device(_batch_data, device)
                        
                        output_dict = model(batch_data)
                        final_loss = criterion(output_dict, _batch_data['ego']['label_dict'])

                    else:
                        batch_data = train_utils.to_device(batch_data, device)
                        batch_data['ego']['epoch'] = epoch

                        if 'vd' in hypes['name']:
                            if hypes['use_camera']:
                                if hypes['use_radar']:
                                    mode = np.array([0, 1, 2])
                                else:
                                    mode = np.array([0, 1])
                            else:
                                mode = np.array([0])
                            output_dict = model(batch_data['ego'], mode=mode)
                        elif 'covqm' in hypes['name']:
                            output_dict = model(batch_data['ego'], training=False)
                        else:
                            output_dict = model(batch_data['ego'])

                        final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                    
                    if False:
                    #if len(output_dict) > 2:
                        single_loss_v = criterion(output_dict, batch_data['ego']['label_dict_single_v'], prefix='_single_v')
                        single_loss_i = criterion(output_dict, batch_data['ego']['label_dict_single_i'], prefix='_single_i')
                        final_loss += single_loss_v + single_loss_i

                        if 'fusion_args' in hypes['model']['args']:
                            if 'communication' in hypes['model']['args']['fusion_args']:
                                comm = hypes['model']['args']['fusion_args']['communication']
                                if ('round' in comm) and comm['round'] > 1:
                                    for round_id in range(1, comm['round']):
                                        round_loss_v = criterion(output_dict, batch_data['ego']['label_dict'], prefix='_v{}'.format(round_id))
                                        final_loss += round_loss_v
                    valid_ave_loss.append(final_loss.item())
                    torch.cuda.empty_cache()

            print("valid_ave_loss: ", valid_ave_loss)
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch, valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            with open(os.path.join(saved_path, 'validation_loss.txt'), 'a+') as f:
                msg = 'Epoch[{}], loss[{}]. \n'.format(epoch+1, valid_ave_loss)
                f.write(msg)

            # lowest val loss
            # if valid_ave_loss < lowest_val_loss:
            #     lowest_val_loss = valid_ave_loss
            #     best_saved_path = os.path.join(saved_path, 'net_epoch_bestval_at{}.pth'.format(epoch+1))
            #     torch.save(model.state_dict(), best_saved_path)

        scheduler.step(epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    run_test = True
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference_heal.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
