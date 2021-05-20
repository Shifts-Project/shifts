# Copyright 2020 The OATomobile Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for logging to Weights & Biases."""

import time

import torch
import wandb
from typing import Dict


class WandbLogger:
    """
    Logs to a `wandb` dashboard.
    Credit to Jannik Kossen @jlko, OATML Group
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def start_counting(self):
        self.train_start = time.time()
        self.checkpoint_start = self.train_start

    def log(
        self, dataset_split_to_loss_dict: Dict[str, Dict[str, torch.Tensor]],
        steps, epoch):
        # Construct loggable dict
        wandb_loss_dict = self.construct_loggable_dict(
            dataset_split_to_loss_dict)
        wandb_loss_dict['step'] = steps
        wandb_loss_dict['epoch'] = epoch
        wandb_loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
        wandb_loss_dict['checkpoint_time'] = (
            f'{time.time() - self.checkpoint_start:.3f}')
        self.checkpoint_start = time.time()

        # Log to wandb
        wandb.log(wandb_loss_dict, step=steps)

    def summary_log(self, loss_dict, new_min):
        # TODO: deal with summary logging with multiple validation sets
        # Do not update summary metrics if not min (min already updated)
        if not new_min:
            return 0

        loss_dict.update({'time': time.time() - self.train_start})

        # Always need to rewrite old summary loss dict, because wandb overrides
        # the summary dict when calling normal log
        lowest_dict = {f'best_{i}': j for i, j in loss_dict.items()}

        wandb.run.summary.update(lowest_dict)

    @staticmethod
    def safe_torch_to_float(val):
        if type(val) == torch.Tensor:
            return val.detach().cpu().numpy().item(0)
        else:
            return val

    @staticmethod
    def construct_loggable_dict(dataset_mode_to_loss_dict):
        wandb_loss_dict = dict()
        for dataset_mode, loss_dict in dataset_mode_to_loss_dict.items():
            for key, value in loss_dict.items():
                key = f'{dataset_mode}_{key}'
                if type(value) == dict:
                    for key2, value2 in value.items():
                        joint_key = f'{key}_{key2}'
                        wandb_loss_dict[joint_key] = (
                            WandbLogger.safe_torch_to_float(value2))
                else:
                    wandb_loss_dict[key] = WandbLogger.safe_torch_to_float(
                        value)

        return wandb_loss_dict

    @staticmethod
    def print_loss_dict(loss_dict):
        train_keys = []
        validation_keys = []
        test_keys = []
        summary_keys = []

        for key in loss_dict.keys():
            if 'train' in key:
                train_keys.append(key)
            elif 'validation' in key:
                validation_keys.append(key)
            elif 'test' in key:
                test_keys.append(key)
            else:
                summary_keys.append(key)

        line = ''
        for key in summary_keys:
            line += f'{key} {loss_dict[key]} | '
        line += f'\nTrain Stats\n'
        for key in train_keys:
            line += f'{key} {loss_dict[key]:.3f} | '
        line += f'\nValidation Stats\n'
        for key in validation_keys:
            line += f'{key} {loss_dict[key]:.3f} | '
        line += f'\nTest Stats\n'
        for key in test_keys:
            line += f'{key} {loss_dict[key]:.3f} | '
        line += '\n'
        print(line)

    # @staticmethod
    # def intermediate_log(loss_dict, num_steps, batch_index, epoch):
    #     """Log during mini-batches."""
    #
    #     tb = 'train_batch'
    #     ld = loss_dict
    #
    #     wandb_dict = dict(
    #         batch_index=batch_index,
    #         epoch=epoch)
    #
    #     losses = dict()
    #
    #     losses.update({f'{tb}_total_loss': ld['total_loss']})
    #
    #     if loss := ld.get('mae_loss', False):
    #         losses.update({f'{tb}_mae_loss': loss})
    #
    #     losses = {i: j.detach().cpu().item() for i, j in losses.items()}
    #     wandb_dict.update(losses)
    #
    #     print(f'step: {num_steps}, {wandb_dict}')
    #     wandb.log(wandb_dict, step=num_steps)
