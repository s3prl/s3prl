import os
import math
import glob
import random
import importlib
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from benchmark.optimizers import get_optimizer


class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, tensorboard logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.logger = SummaryWriter(args.expdir)
        
        self.init_ckpt = torch.load(self.args.past_exp, map_location='cpu') if self.args.past_exp else {}

        module_path = f'benchmark.upstream.{args.upstream}.expert'
        Upstream = getattr(importlib.import_module(module_path), 'UpstreamExpert')
        self.upstream = Upstream(
            self.args.upstream_ckpt,
            self.args.upstream_config
        ).to(self.args.device)

        if init_upstream := self.init_ckpt.get('Upstream'):
            self.upstream.load_state_dict(init_upstream)

        module_path = f'benchmark.downstream.{args.downstream}.expert'
        Downstream = getattr(importlib.import_module(module_path), 'DownstreamExpert')
        self.downstream = Downstream(
            self.upstream.get_output_dim(),
            **vars(self.args),
            **self.config['expert']
        ).to(self.args.device)

        if init_downstream := self.init_ckpt.get('Downstream'):
            self.downstream.load_state_dict(init_downstream)


    def train(self):
        # set model train/eval modes
        self.downstream.train()
        self.upstream.eval()
        if self.args.upstream_trainable:
            self.upstream.train()

        # choose models to optimize
        optimized_models = [self.downstream]
        if self.args.upstream_trainable:
            optimized_models.append(self.upstream)

        # set optimizer
        optimizer = get_optimizer(optimized_models, **self.config['optimizer'])
        if init_optimizer := self.init_ckpt.get('Optimizer'):
            optimizer.load_state_dict(init_optimizer)

        # set progress bar
        pbar = tqdm(total=self.config['optimizer']['total_steps'], desc='overall')
        if init_step := self.init_ckpt.get('Step'):
            pbar.n = init_step - 1

        if self.args.eval_init:
            for split in self.config['runner']['eval_splits']:
                self.evaluate(split, pbar.n)

        # prepare data
        dataloader = self.downstream.get_train_dataloader()

        all_loss = []
        records = defaultdict(list)
        while pbar.n < pbar.total:
            for wavs, *others in tqdm(dataloader, dynamic_ncols=True, desc='train'):
                try:
                    if pbar.n >= pbar.total:
                        break
                    
                    wavs = [wav.to(self.args.device) for wav in wavs]
                    if self.args.upstream_trainable:
                        features = self.upstream(wavs)
                    else:
                        with torch.no_grad():
                            features = self.upstream(wavs)

                    loss = self.downstream(features, *others, records=records, logger=self.logger, global_step=pbar.n)
                    loss.backward()

                    # record loss
                    all_loss.append(loss.item())

                    # gradient clipping
                    paras = list(self.downstream.parameters())
                    if self.args.upstream_trainable:
                        paras += list(self.upstream.parameters())
                    grad_norm = torch.nn.utils.clip_grad_norm_(paras, self.config['optimizer']['gradient_clipping'])

                    # optimize
                    if math.isnan(grad_norm):
                        print(f'[Runner] - Error : grad norm is NaN at step {pbar.n}')
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    # logging
                    if (pbar.n + 1) % self.config['runner']['log_step'] == 0:
                        average_loss = torch.FloatTensor(all_loss).mean().item()
                        self.logger.add_scalar(f'{self.args.downstream}/train-loss', average_loss, global_step=pbar.n)
                        for key, values in records.items():
                            average = torch.FloatTensor(values).mean().item()
                            self.logger.add_scalar(f'{self.args.downstream}/train-{key}', average, global_step=pbar.n)
                        all_loss = []
                        records = defaultdict(list)

                    # evaluation
                    if (pbar.n + 1) % self.config['runner']['eval_step'] == 0:
                        for split in self.config['runner']['eval_splits']:
                            self.evaluate(split, pbar.n)                        

                    # save checkpoint
                    if (pbar.n + 1) % self.config['runner']['save_step'] == 0:
                        all_states = {
                            'Downstream': self.downstream.state_dict(),
                            'Optimizer': optimizer.state_dict(),
                            'Step': pbar.n,
                            'Args': self.args,
                            'Config': self.config,
                        }
                        if self.args.upstream_trainable:
                            all_states['Upstream'] = self.upstream.state_dict()

                        def check_ckpt_num(directory):
                            max_keep = self.config['runner']['max_keep']
                            ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                            if len(ckpt_pths) >= max_keep:
                                ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                                for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                    os.remove(ckpt_pth)

                        check_ckpt_num(self.args.expdir)
                        torch.save(all_states, f'{self.args.expdir}/states-{pbar.n}.ckpt')                        

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {pbar.n}')
                    else:
                        raise
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()

                pbar.update(1)

        pbar.close()


    def evaluate(self, split='test', global_step=0):
        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        downstream_training = self.downstream.training
        upstream_training = self.upstream.training
        self.downstream.eval()
        self.upstream.eval()

        # prepare data
        dataloader = eval(f'self.downstream.get_{split}_dataloader')()

        # main evaluation block
        all_loss = []
        records = defaultdict(list)
        for wavs, *others in tqdm(dataloader, dynamic_ncols=True, desc=split):

            wavs = [wav.to(self.args.device) for wav in wavs]
            with torch.no_grad():
                features = self.upstream(wavs)

            loss = self.downstream(features, *others, records=records, logger=self.logger, global_step=global_step)
            all_loss.append(loss.item())
        
        # logging
        average_loss = torch.FloatTensor(all_loss).mean().item()
        self.logger.add_scalar(f'{self.args.downstream}/{split}-loss', average_loss, global_step=global_step)
        for key, values in records.items():
            average = torch.FloatTensor(values).mean().item()
            self.logger.add_scalar(f'{self.args.downstream}/{split}-{key}', average, global_step=global_step)

        # prepare back to training
        torch.cuda.empty_cache()
        if downstream_training:
            self.downstream.train()
        if upstream_training:
            self.upstream.train()
