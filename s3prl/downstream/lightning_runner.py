import os
import sys
import math
import glob
import uuid
import shutil
import random
import tempfile
import importlib
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from s3prl import hub
from s3prl import downstream
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict
from s3prl.utility.lightning.callbacks import LitProgressBar, Saver

import pytorch_lightning as pl

from huggingface_hub import HfApi, HfFolder, Repository

SAMPLE_RATE = 16000


class ModelEntry(pl.LightningModule):
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces

        if not self.trainable:
            self.freeze()


class Runner(pl.LightningModule):
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
        self.downstream = self._get_downstream()
        self.all_entries = [self.upstream, self.featurizer, self.downstream]

        # specaug
        self.specaug = None
        if self.config.get('specaug'):
            from .specaug import SpecAug
            self.specaug = SpecAug(**self.config["specaug"])


    @staticmethod
    def add_model_specific_args(parent_parser):
        # NOTE: TODO
        parser = parent_parser.add_argument_group("LitMNIST")
        parser.add_argument("--layer_1_dim", type=int, default=128)
        return parent_parser


    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)


    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface)

        self._load_weight(model, name)

        return ModelEntry(model, name, trainable, interfaces)


    def _get_upstream(self):
        if "from_hf_hub" in self.args and self.args.from_hf_hub == True:
            from huggingface_hub import snapshot_download

            print(f'Downloading upstream model {self.args.upstream} from the Hugging Face Hub')
            filepath = snapshot_download(self.args.upstream)
            sys.path.append(filepath)

            from expert import UpstreamExpert
            Upstream = UpstreamExpert
            ckpt_path = os.path.join(filepath, self.args.upstream_model_name)
        else:
            Upstream = getattr(hub, self.args.upstream)
            ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        )

        return self._init_model(
            model = model,
            name = 'Upstream',
            trainable = self.args.upstream_trainable,
        )


    def _get_featurizer(self):
        model = Featurizer(
            self.upstream.model, self.args.upstream_feature_selection,
            upstream_device=self.args.device,
        )

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate']
        )


    def _get_downstream(self):
        Downstream = getattr(downstream, self.args.downstream)
        model = Downstream(
            upstream_dim = self.featurizer.model.output_dim,
            upstream_rate = self.featurizer.model.downsample_rate,
            **self.config,
            **vars(self.args)
        )

        return self._init_model(
            model = model,
            name = 'Downstream',
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


    def configure_optimizers(self):
        # trainable parameters and train/eval mode
        trainable_models = []
        for entry in self.all_entries:
            if entry.trainable:
                trainable_models.append(entry.model)

        # optimizer
        optimizer = self._get_optimizer(trainable_models)

        # scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = {
                'scheduler': self._get_scheduler(optimizer),
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [scheduler]

        return optimizer


    def configure_callbacks(self):
        callbacks = [Saver()]
        callbacks.append(LitProgressBar())  # NOTE: optional
        return callbacks


    def train_dataloader(self):
        self.train_dataloader = self.downstream.model.get_dataloader('train')
        return self.train_dataloader


    def transfer_batch_to_device(self, batch, device):
        # Overwrite lightning default transfer options to s3prl settings
        wavs, *others = batch
        wavs = [torch.FloatTensor(wav).to(device) for wav in wavs]
        batch = (wavs, *others)
        return batch


    def on_train_epoch_start(self):
        for entry in self.all_entries:
            if not entry.trainable:
                entry.eval()


    def training_step(self, batch, batch_idx):
        wavs, *others = batch

        with torch.set_grad_enabled(self.upstream.trainable):
            features = self.upstream.model(wavs)
        features = self.featurizer.model(wavs, features)
        if self.specaug:
            features, _ = self.specaug(features)

        records = defaultdict(list)
        loss = self.downstream.model(
            'train',
            features, *others,
            records = records,
        )

        return {'loss': loss, 'records': records}


    def validation_step(self, batch, batch_idx, dataloader_idx):
        wavs, *others = batch

        features = self.upstream.model(wavs)
        features = self.featurizer.model(wavs, features)
        if self.specaug:
            features, _ = self.specaug(features)

        split = self.config['runner']['eval_dataloaders'][dataloader_idx]
        records = defaultdict(list)
        self.downstream.model(
            split,
            features, *others,
            records = records,
            batch_id = batch_idx,
        )

        return {'records': records}


    def on_save_checkpoint(self, checkpoint):
        # The following are automatically saved in checkpoint by lightning:
        # - 16-bit scaling factor (apex)
        # - Current epoch, global step
        # - Model state_dict, optimizer states, scheduler states, callback states
        # - Hyperparameters passed to the model (Argparse.Namespace)
        checkpoint.update({
            'Args': self.args,
            'Config': self.config,
        })

    def on_load_checkpoint(self, checkpoint):
        self.args = checkpoint['Args']
        self.config = checkpoint['Config']
