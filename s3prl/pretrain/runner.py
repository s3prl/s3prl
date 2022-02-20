# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ run_pretrain.py ]
#   Synopsis     [ scripts for running the pre-training of upstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import glob
import random
import importlib
from tqdm import tqdm
from collections import defaultdict
#-------------#
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
#-------------#
from optimizers import get_optimizer, get_grouped_parameters
from schedulers import get_scheduler


##########
# RUNNER #
##########
class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, tensorboard logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.logger = SummaryWriter(args.expdir)                                                 

        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        self.upstream = self._get_upstream()


    def _get_upstream(self):
        init_upstream = self.init_ckpt.get('Upstream_Config')
        if init_upstream:
            self.args.upstream_config = init_upstream
        module_path = f'pretrain.{self.args.upstream}.pretrain_expert'
        Upstream = getattr(importlib.import_module(module_path), 'UpstreamPretrainExpert')
        upstream = Upstream(self.config['pretrain_expert']['datarc'], 
                            self.args.upstream_config,
                            self.args.device,
                            self.args.multi_gpu).to(self.args.device)

        assert hasattr(upstream, 'device')
        assert hasattr(upstream, 'forward')
        assert hasattr(upstream, 'load_model')
        assert hasattr(upstream, 'add_state_to_save')
        assert hasattr(upstream, 'on_before_zero_grad')
        assert hasattr(upstream, 'get_train_dataloader')

        if self.init_ckpt != {}:
            print('[Runner] - Loading upstream weights from the previous experiment')
            upstream.load_model(self.init_ckpt)
        if hasattr(upstream, 'loss_to_device'):
            print('[Runner] - Loss to device')
            upstream.loss_to_device()
        return upstream


    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )

        if self.init_ckpt != {}:
            init_optimizer = self.init_ckpt.get('Optimizer')
            assert init_optimizer
            print('[Runner] - Loading optimizer weights from the previous experiment')
            optimizer.load_state_dict(init_optimizer)
        return optimizer


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )

        if self.init_ckpt != {}:
            init_scheduler = self.init_ckpt.get('Scheduler')
            assert init_scheduler
            print('[Runner] - Loading scheduler weights from the previous experiment')
            scheduler.load_state_dict(init_scheduler)
        return scheduler


    def train(self):
        # set model train mode
        self.upstream.train()

        # prepare data
        gradient_accumulate_steps = self.config['runner']['gradient_accumulate_steps']
        train_batch_size = self.config['pretrain_expert']['datarc']['train_batch_size']
        print('[Runner] - Accumulated batch size:', train_batch_size * gradient_accumulate_steps)
        dataloader = self.upstream.get_train_dataloader()

        # set epoch
        n_epochs = self.config['runner']['n_epochs']
        if n_epochs > 0: 
            total_steps = int(n_epochs * len(dataloader.dataset) / gradient_accumulate_steps)
            print(f'[Runner] - Training for {n_epochs} epochs, which is equivalent to {total_steps} steps')
        else:
            total_steps = self.config['runner']['total_steps']
            n_epochs = int(total_steps * gradient_accumulate_steps / len(dataloader.dataset))
            print(f'[Runner] - Training for {total_steps} steps, which is approximately {n_epochs} epochs')

        assert total_steps > self.config['runner']['log_step']
        assert total_steps > self.config['runner']['save_step']

        # set amp
        amp = self.config['runner'].get('fp16', False)
        if amp:
            print('[Runner] - Enabled fp16 training')
            scaler = torch.cuda.amp.GradScaler()

        # set optimizer
        model_params = [self.upstream.model]
        optimizer = self._get_optimizer(model_params)

        # set scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)

        # set progress bar
        pbar = tqdm(total=total_steps, dynamic_ncols=True, desc='overall')
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        all_loss = 0
        backward_steps = 0
        records = defaultdict(list)
        prefix = f'{self.args.upstream}/train-'

        while pbar.n < pbar.total:
            for data in tqdm(dataloader, dynamic_ncols=True, desc='train'):
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    with torch.cuda.amp.autocast(enabled=amp):
                        loss, records = self.upstream(
                            data,
                            records=records,
                            global_step=global_step,
                            log_step=self.config['runner']['log_step'],
                        )

                    if gradient_accumulate_steps > 1:
                        loss = loss / gradient_accumulate_steps
                    if self.args.multi_gpu:
                        loss = loss.sum()
                    if amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # record loss
                all_loss += loss.item()
                del loss
                
                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue
                    
                # unscale
                if amp:
                    scaler.unscale_(optimizer)

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.upstream.model.parameters(), self.config['runner']['gradient_clipping'])
                if math.isnan(grad_norm):
                    print(f'[Runner] - Error : grad norm is NaN at global step {global_step}')

                # optimize
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                elif not math.isnan(grad_norm):
                    optimizer.step()

                self.upstream.on_before_zero_grad()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()

                # logging
                if global_step % self.config['runner']['log_step'] == 0 or pbar.n == pbar.total -1:
                    # log loss
                    self.logger.add_scalar(f'{prefix}loss', all_loss, global_step=global_step)
                    # log lr
                    if hasattr(optimizer, 'get_lr'):
                        self.logger.add_scalar(f'{prefix}lr', optimizer.get_lr()[0], global_step=global_step)
                    else:
                        self.logger.add_scalar(f'{prefix}lr', self.config['optimizer']['lr'], global_step=global_step)
                    # log norm
                    self.logger.add_scalar(f'{prefix}gradient-norm', grad_norm, global_step=global_step)

                    # log customized contents
                    self.upstream.log_records(
                        records=records,
                        logger=self.logger,
                        prefix=prefix,
                        global_step=global_step,
                    )
                    records = defaultdict(list)

                if global_step % self.config['runner']['save_step'] == 0 or pbar.n == pbar.total -1:
                    def check_ckpt_num(directory):
                        max_keep = self.config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)

                    all_states = {
                        'Optimizer': optimizer.state_dict(),
                        'Step': pbar.n,
                        'Args': self.args,
                        'Config': self.config,
                    }
                    all_states = self.upstream.add_state_to_save(all_states)

                    if scheduler:
                        all_states['Scheduler'] = scheduler.state_dict()
                    
                    name = f'states-epoch-{n_epochs}.ckpt' if pbar.n == pbar.total -1 and n_epochs > 0 else \
                           f'states-{global_step}.ckpt'
                    save_path = os.path.join(self.args.expdir, name)
                    tqdm.write(f'[Runner] - Save the checkpoint to: {save_path}')
                    torch.save(all_states, save_path)
                
                all_loss = 0      
                pbar.update(1)

        pbar.close()
