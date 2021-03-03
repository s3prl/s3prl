import os
import sys
import math
import glob
import random
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

from optimizers import get_optimizer
from schedulers import get_scheduler
from utility.helper import is_leader_process, count_parameters, get_model_state, show, defaultdict

SAMPLE_RATE = 16000


class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        self.upstream = self._get_upstream()
        self.downstream = self._get_downstream()

        if is_leader_process():
            self.logger = SummaryWriter(args.expdir)


    def _get_upstream(self):
        Upstream = getattr(importlib.import_module('hubconf'), self.args.upstream)
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        upstream = Upstream(
            feature_selection = self.args.upstream_feature_selection,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
            ckpt = self.args.upstream_ckpt,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        interface_fn = ['get_output_dim', 'get_downsample_rate']
        for fn in interface_fn:
            assert hasattr(upstream, fn)

        show(f'[Runner] - Upstream model architecture: {upstream}')
        show(f'[Runner] - Upstream output dimension: {upstream.get_output_dim()}')
        downsample = upstream.get_downsample_rate()
        show(f'[Runner] - Upstream downsample rate: {downsample} ({downsample / SAMPLE_RATE * 1000} ms/frame)')

        init_upstream = self.init_ckpt.get('Upstream')
        if init_upstream:
            show('[Runner] - Loading upstream weights from the previous experiment')
            upstream.load_state_dict(init_upstream)

        if is_initialized() and self.args.upstream_trainable:
            upstream = DDP(upstream, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for fn in interface_fn:
                setattr(upstream, fn, getattr(upstream.module, fn))

        return upstream


    def _get_downstream(self):
        module_path = f'downstream.{self.args.downstream}.expert'
        Downstream = getattr(importlib.import_module(module_path), 'DownstreamExpert')
        downstream = Downstream(
            upstream_dim = self.upstream.get_output_dim(),
            upstream_rate = self.upstream.get_downsample_rate(),
            **self.config,
            **vars(self.args)
        ).to(self.args.device)

        show(f'[Runner] - Downstream model architecture: {downstream}')
        show(f'[Runner] - Downstream has {count_parameters(downstream)} parameters')

        interface_fn = ['get_dataloader', 'log_records']
        for fn in interface_fn:
            assert hasattr(downstream, fn)

        init_downstream = self.init_ckpt.get('Downstream')
        if init_downstream:
            show('[Runner] - Loading downstream weights from the previous experiment')
            downstream.load_state_dict(init_downstream)

        if is_initialized():
            downstream = DDP(downstream, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for fn in interface_fn:
                setattr(downstream, fn, getattr(downstream.module, fn))

        return downstream


    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )

        init_optimizer = self.init_ckpt.get('Optimizer')
        if init_optimizer:
            show('[Runner] - Loading optimizer weights from the previous experiment')
            optimizer.load_state_dict(init_optimizer)
        return optimizer


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )

        init_scheduler = self.init_ckpt.get('Scheduler')
        if init_scheduler:
            show('[Runner] - Loading scheduler weights from the previous experiment')
            scheduler.load_state_dict(init_scheduler)
        return scheduler


    def train(self):
        # set model train/eval modes
        self.downstream.train()
        self.upstream.eval()
        if self.args.upstream_trainable:
            self.upstream.train()

        # set optimizer
        model_params = [self.downstream]
        if self.args.upstream_trainable:
            model_params.append(self.upstream)
        optimizer = self._get_optimizer(model_params)

        # set scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)

        # set specaug
        specaug = None
        if self.config.get('specaug'):
            from .specaug import SpecAug
            specaug = SpecAug(**self.config["specaug"])

        # set progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        pbar = tqdm(total=self.config['runner']['total_steps'], dynamic_ncols=True, desc='overall', file=tqdm_file)
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        # prepare data
        dataloader = self.downstream.get_dataloader('train')

        batch_ids = []
        backward_steps = 0
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', 0)
        while pbar.n < pbar.total:
            if is_initialized():
                dataloader.sampler.set_epoch(epoch)

            for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc='train', file=tqdm_file)):
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    wavs = [wav.to(self.args.device) for wav in wavs]
                    if self.upstream.training:
                        features = self.upstream(wavs)
                    else:
                        with torch.no_grad():
                            features = self.upstream(wavs)

                    if specaug:
                        features, _ = specaug(features)

                    loss = self.downstream(
                        'train',
                        features, *others,
                        records = records,
                    )
                    batch_ids.append(batch_id)

                    gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                    (loss / gradient_accumulate_steps).backward()
                    del loss

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        if is_initialized():
                            raise
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                # gradient clipping
                paras = list(self.downstream.parameters())
                if self.args.upstream_trainable:
                    paras += list(self.upstream.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(paras, self.config['runner']['gradient_clipping'])

                # optimize
                if math.isnan(grad_norm):
                    print(f'[Runner] - grad norm is NaN at step {global_step}')
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()

                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    continue

                # logging
                if global_step % self.config['runner']['log_step'] == 0:
                    self.downstream.log_records(
                        'train',
                        records = records,
                        logger = self.logger,
                        global_step = global_step,
                        batch_ids = batch_ids,
                        total_batch_num = len(dataloader),
                    )
                    batch_ids = []
                    records = defaultdict(list)

                # evaluation and save checkpoint
                save_names = []

                if global_step % self.config['runner']['eval_step'] == 0:
                    for split in self.config['runner']['eval_dataloaders']:
                        save_names += self.evaluate(split, global_step)

                if global_step % self.config['runner']['save_step'] == 0:
                    def check_ckpt_num(directory):
                        max_keep = self.config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)
                    save_names.append(f'states-{global_step}.ckpt')

                if len(save_names) > 0:
                    all_states = {
                        'Downstream': get_model_state(self.downstream),
                        'Optimizer': optimizer.state_dict(),
                        'Step': global_step,
                        'Epoch': epoch,
                        'Args': self.args,
                        'Config': self.config,
                    }

                    if scheduler:
                        all_states['Scheduler'] = scheduler.state_dict()

                    if self.args.upstream_trainable:
                        all_states['Upstream'] = get_model_state(self.upstream)

                    save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                    tqdm.write(f'[Runner] - Save the checkpoint to:')
                    for i, path in enumerate(save_paths):
                        tqdm.write(f'{i + 1}. {path}')
                        torch.save(all_states, path)

                pbar.update(1)
            epoch += 1
        pbar.close()


    def evaluate(self, split=None, global_step=0):
        split = split or self.args.evaluate_split

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
        dataloader = self.downstream.get_dataloader(split)

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split)):

            wavs = [wav.to(self.args.device) for wav in wavs]
            with torch.no_grad():
                features = self.upstream(wavs)
                self.downstream(
                    split,
                    features, *others,
                    records = records,
                )
                batch_ids.append(batch_id)

        save_names = self.downstream.log_records(
            split,
            records = records,
            logger = self.logger,
            global_step = global_step,
            batch_ids = batch_ids,
            total_batch_num = len(dataloader),
        )
        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        torch.cuda.empty_cache()
        if downstream_training:
            self.downstream.train()
        if upstream_training:
            self.upstream.train()

        return [] if type(save_names) is not list else save_names
