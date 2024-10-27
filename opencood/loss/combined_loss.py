import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from importlib import import_module


class CombinedLoss(nn.Module):
    def __init__(self, args):
        super(CombinedLoss, self).__init__()
        self.losses = []
        for k, v in args.items():
            module = import_module(f"opencood.loss.{k}")
            loss_name = ''.join([x[0].upper() + x[1:] for x in k.split('_')])
            self.losses.append(getattr(module, loss_name)(v))
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, prefix=''):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        total_loss = 0
        for loss in self.losses:
            total_loss += loss(output_dict, target_dict, prefix)
            self.loss_dict.update(loss.loss_dict)
        self.loss_dict['total_loss'] = total_loss

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']
        static_loss = self.loss_dict.get('static_loss', None)

        total_loss_ego = self.loss_dict.get('total_loss_ego', torch.tensor(0.0))
        reg_loss_ego = self.loss_dict.get('reg_loss_ego', torch.tensor(0.0))
        conf_loss_ego = self.loss_dict.get('conf_loss_ego', torch.tensor(0.0))

        if static_loss is None:
            static_loss = torch.tensor(0.0)
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f "
                  "|| Dynamic Loss: %.4f || Static Loss: %.4f "
                  "|| Conf Loss: %.4f || Loc Loss: %.4f "
                " || Ego Loss: %.4f, %.4f, %.4f" % (
                    epoch, batch_id + 1, batch_len, total_loss.item(),
                    dynamic_loss.item(), static_loss.item(),
                    conf_loss.item(), reg_loss.item(),
                    total_loss_ego.item(), reg_loss_ego.item(), conf_loss_ego.item()
                )
            )
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f "
                                 "|| Dynamic Loss: %.4f "
                                 "|| Static Loss: %.4f "
                                 "|| Conf Loss: %.4f"
                                 " || Loc Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len, total_loss.item(),
                      dynamic_loss.item(), static_loss.item(),
                      conf_loss.item(), reg_loss.item()))


        writer.add_scalar('Regression_loss', reg_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Confidence_loss', conf_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Dynamic_loss', dynamic_loss.item(),
                          epoch*batch_len + batch_id)
        if static_loss > 0:
            writer.add_scalar('Static_loss', static_loss.item(),
                              epoch*batch_len + batch_id)
        writer.add_scalar('Total_loss', total_loss.item(),
                          epoch*batch_len + batch_id)
