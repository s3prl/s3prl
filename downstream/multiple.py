import os
import sys
import math
import glob
import shutil
import random
import tempfile
import importlib

import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

import hubconf
from optimizers import get_optimizer
from schedulers import get_scheduler
from upstream.interfaces import Featurizer
from utility.helper import is_leader_process, get_model_state, show, defaultdict

SAMPLE_RATE = 16000


class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class DataEntry:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = -1
        self.records = defaultdict(list)
        self.batch_ids = []

    @staticmethod
    def make_iter(dataloader, epoch):
        if is_initialized():
            dataloader.sampler.set_epoch(epoch)
        return iter(dataloader)

    def next_batch(self):
        try:
            current_batch = next(self._iterator)
        except (StopIteration, AttributeError) as e:
            self.epoch += 1
            self._iterator = self.make_iter(self.dataloader, self.epoch)
            self._next_batch_id = 0
            current_batch = next(self._iterator)

        batch_id = self._next_batch_id
        self._next_batch_id += 1
        return batch_id, current_batch


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
        self.featurizer = self._get_featurizer()

        self.downstreams = [self._get_downstream(d) for d in self.args.downstream.split(",")]
        self.all_models = [self.upstream, self.featurizer] + self.downstreams


    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)


    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface)

        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)


    def _get_upstream(self):
        Upstream = getattr(hubconf, self.args.upstream)
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt = self.args.upstream_ckpt,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model = model,
            name = 'Upstream',
            trainable = self.args.upstream_trainable,
        )


    def _get_featurizer(self):
        model = Featurizer(
            self.upstream.model, self.args.upstream_feature_selection
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate']
        )


    def _get_downstream(self, downstream: str):
        module_path = f'downstream.{downstream}.expert'
        Downstream = getattr(importlib.import_module(module_path), 'DownstreamExpert')
        
        expdir = f"{self.args.expdir}/{downstream}"
        os.makedirs(expdir, exist_ok=True)
        
        model = Downstream(
            upstream_dim = self.featurizer.model.output_dim,
            upstream_rate = self.featurizer.model.downsample_rate,
            downstream_expert = self.config[downstream],
            expdir = expdir,
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = downstream,
            trainable = True,
            interfaces = ['get_dataloader', 'log_records']
        )


    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )
        self._load_weight(optimizer, 'Optimizer')
        return optimizer


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )
        self._load_weight(scheduler, 'Scheduler')
        return scheduler


    def train(self):
        # trainable parameters and train/eval mode
        trainable_models = []
        trainable_paras = []
        for entry in self.all_models:
            if entry.trainable:
                entry.model.train()
                trainable_models.append(entry.model)
                trainable_paras += list(entry.model.parameters())
            else:
                entry.model.eval()

        # optimizer
        optimizer = self._get_optimizer(trainable_models)

        # scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)

        # specaug
        specaug = None
        if self.config.get('specaug'):
            from .specaug import SpecAug
            specaug = SpecAug(**self.config["specaug"])

        # progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        pbar = tqdm(total=self.config['runner']['total_steps'], dynamic_ncols=True, desc='overall', file=tqdm_file)
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)

        # prepare data
        data_entries = []
        for expert in self.downstreams:
            dataloader = expert.model.get_dataloader("train")
            data_entry = DataEntry(dataloader)
            init_epoch = self.init_ckpt.get(f"Epoch.{expert.name}")
            if init_epoch:
                data_entry.epoch = init_epoch
            data_entries.append(data_entry)

        backward_steps = 0
        while pbar.n < pbar.total:
            loss = 0
            for expert, data in zip(self.downstreams, data_entries):
                batch_id, (wavs, *others) = data.next_batch()
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                    if self.upstream.trainable:
                        features = self.upstream.model(wavs)
                    else:
                        with torch.no_grad():
                            features = self.upstream.model(wavs)
                    features = self.featurizer.model(wavs, features)

                    if specaug:
                        features, _ = specaug(features)

                    loss += expert.model(
                        'train',
                        features, *others,
                        records = data.records,
                    )
                    data.batch_ids.append(batch_id)

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        if is_initialized():
                            raise
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

            gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
            (loss / gradient_accumulate_steps).backward()
            del loss

            # whether to accumulate gradient
            backward_steps += 1
            if backward_steps % gradient_accumulate_steps > 0:
                continue

            # gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_paras, self.config['runner']['gradient_clipping'])

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
                data.batch_ids = []
                data.records = defaultdict(list)
                continue

            # evaluation and save checkpoint
            save_names = []
            for expert, data in zip(self.downstreams, data_entries):
                looprc = self.config[expert.name]['looprc']

                if global_step % looprc['log_step'] == 0:
                    expert.model.log_records(
                        'train',
                        records = data.records,
                        logger = logger,
                        global_step = global_step,
                        batch_ids = data.batch_ids,
                        total_batch_num = len(data.dataloader),
                    )
                    data.batch_ids = []
                    data.records = defaultdict(list)

                if global_step % looprc['eval_step'] == 0:
                    for split in looprc['eval_dataloaders']:
                        save_names_per_task = self.evaluate(f"{expert.name}:{split}", logger, global_step)
                        save_names += [f"{expert.name}/{n}" for n in save_names_per_task]

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
                    'Optimizer': optimizer.state_dict(),
                    'Step': global_step,
                    'Args': self.args,
                    'Config': self.config,
                }

                for entry in self.all_models:
                    if entry.trainable:
                        all_states[entry.name] = get_model_state(entry.model)

                for entry, expert in zip(data_entries, self.downstreams):
                    all_states[f"Epoch.{expert.name}"] = entry.epoch

                if scheduler:
                    all_states['Scheduler'] = scheduler.state_dict()

                if is_initialized():
                    all_states['WorldSize'] = get_world_size()

                save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                tqdm.write(f'[Runner] - Save the checkpoint to:')
                for i, path in enumerate(save_paths):
                    tqdm.write(f'{i + 1}. {path}')
                    torch.save(all_states, path)

            pbar.update(1)

        pbar.close()
        if is_leader_process():
            logger.close()


    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.cuda.device(self.args.device):
            torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in self.all_models:
            trainings.append(entry.model.training)
            entry.model.eval()

        downstream, split = split.split(":")
        downstream = [d for d in self.downstreams if d.name == downstream][0]
        dataloader = downstream.model.get_dataloader(split)

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=f"{downstream.name}-{split}")):

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            with torch.no_grad():
                features = self.upstream.model(wavs)
                features = self.featurizer.model(wavs, features)
                downstream.model(
                    split,
                    features, *others,
                    records = records,
                )
                batch_ids.append(batch_id)

        save_names = downstream.model.log_records(
            split,
            records = records,
            logger = logger,
            global_step = global_step,
            batch_ids = batch_ids,
            total_batch_num = len(dataloader),
        )
        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        with torch.cuda.device(self.args.device):
            torch.cuda.empty_cache()

        for entry, training in zip(self.all_models, trainings):
            if training:
                entry.model.train()

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        return [] if type(save_names) is not list else save_names
