import os
import sys
import math
import glob
import uuid
import shutil
import logging
import tempfile
import importlib
from typing import Any
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

from s3prl import hub
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, defaultdict
from huggingface_hub import HfApi, HfFolder, Repository
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_CONTINUAL_ERROR = 20
ACCEPTABLE_ERRORS = [
    "CUDA out of memory",
    "Unable to find a valid cuDNN algorithm to run convolution",  # Usually caused by CUDA OOM
]
MODEL_CARD_MARKDOWN = """---
datasets:
- superb
tags:
- library:s3prl
- benchmark:superb
- type:model
---

# Fine-tuned s3prl model

Upstream Model: {upstream_model}

## Model description

[More information needed]

## Intended uses & limitations

[More information needed]

## How to use

[More information needed]

## Limitations and bias

[More information needed]

## Training data

[More information needed]

## Training procedure

[More information needed]

## Evaluation results

[More information needed]

"""


class ModelEntry:
    def __init__(self, name, model, trainable):
        self.name = name
        self.model = model
        self.local_model = model.module if isinstance(model, DDP) else model
        self.trainable = trainable

        paras = list(self.local_model.parameters())
        self.device = None if len(paras) == 0 else paras[0].device

    def __getattr__(self, __name: str) -> Any:
        if hasattr(self.model, __name):
            return getattr(self.model, __name)
        else:
            return getattr(self.local_model, __name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.device is not None:
            self.model.to(self.device)

        if torch.is_grad_enabled() and self.trainable:
            self.model.train()
        else:
            self.model.eval()

        if torch.is_grad_enabled():
            return self.model(*args, **kwargs)
        else:
            return self.local_model(*args, **kwargs)


class Runner:
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


    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            log.info(f'Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)


    def _init_model(self, name, model, trainable):
        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            local_rank = int(os.environ.get("LOCAL_RANK"))
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        return ModelEntry(name, model, trainable)


    def _get_upstream(self):
        if "from_hf_hub" in self.args and self.args.from_hf_hub == True:
            from huggingface_hub import snapshot_download

            log.info(f'Downloading upstream model {self.args.upstream} from the Hugging Face Hub')
            filepath = snapshot_download(self.args.upstream, self.args.upstream_revision, use_auth_token=True)
            sys.path.append(filepath)

            dependencies = (Path(filepath) / 'requirements.txt').resolve()
            log.info("The downloaded upstream model requires the following dependencies. Please make sure they are installed:")
            for idx, line in enumerate((Path(filepath) / "requirements.txt").open().readlines()):
                log.info(f"{idx}. {line.strip()}")
            log.info(f"You can install them by:")
            log.info()
            log.info(f"pip install -r {dependencies}")
            log.info()

            from expert import UpstreamExpert
            Upstream = UpstreamExpert
            ckpt_path = os.path.join(filepath, self.args.upstream_model_name)
        else:
            Upstream = getattr(hub, self.args.upstream)
            ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
            **self.config.get("upstream_expert", {}),
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            name = 'Upstream',
            model = model,
            trainable = self.args.upstream_trainable,
        )


    def _get_featurizer(self):
        model = Featurizer(
            upstream = self.upstream.model,
            feature_selection = self.args.upstream_feature_selection,
            layer_selection = self.args.upstream_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
        ).to(self.args.device)

        return self._init_model(
            name = 'Featurizer',
            model = model,
            trainable = True,
        )


    def _get_downstream(self):
        expert = importlib.import_module(f"s3prl.downstream.{self.args.downstream}.expert")
        Downstream = getattr(expert, "DownstreamExpert")

        model = Downstream(
            upstream_dim = self.featurizer.output_dim,
            upstream_rate = self.featurizer.downsample_rate,
            **self.config,
            **vars(self.args)
        ).to(self.args.device)

        return self._init_model(
            name = 'Downstream',
            model = model,
            trainable = True,
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

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(upstream_model=self.args.upstream)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)


    def train(self):
        # trainable parameters and train/eval mode
        trainable_models = []
        trainable_paras = []
        for entry in self.all_entries:
            if entry.trainable:
                trainable_models.append(entry.model)
                trainable_paras += list(entry.model.parameters())

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
        optimization_steps = round(self.config['runner']['total_steps'] * self.config["runner"].get("optimize_ratio", 1))
        pbar = tqdm(total=optimization_steps, dynamic_ncols=True, desc='overall', file=tqdm_file)
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)

        def save_states(global_step, epoch, filenames):
            all_states = {
                'Optimizer': optimizer.state_dict(),
                'Step': global_step,
                'Epoch': epoch,
                'Args': self.args,
                'Config': self.config,
            }

            for entry in self.all_entries:
                if entry.trainable:
                    all_states[entry.name] = get_model_state(entry.model)

            if scheduler:
                all_states['Scheduler'] = scheduler.state_dict()

            if is_initialized():
                all_states['WorldSize'] = get_world_size()

            filenames = list(set(filenames))
            save_paths = [os.path.join(self.args.expdir, name) for name in filenames]
            log.info("Save the checkpoint to: (checkpoints will only be saved by rank 0 during DDP training)")
            for i, path in enumerate(save_paths):
                log.info(f'{i + 1}. {path}')
                torch.save(all_states, path)

        batch_ids = []
        backward_steps = 0
        continual_error = 0
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', -1)
        train_split = self.config['runner'].get("train_dataloader", "train")
        gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps', 1)
        training_completed = False
        while not training_completed:
            epoch += 1
            try:
                dataloader = self.downstream.get_dataloader(train_split, epoch=epoch)
            except TypeError as e:
                if "unexpected keyword argument 'epoch'" in str(e):
                    dataloader = self.downstream.get_dataloader(train_split)
                    if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
                        dataloader.sampler.set_epoch(epoch)
                else:
                    raise

            log.info(f"Start training epoch {epoch}...")
            for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc='train', file=tqdm_file)):
                # try/except block for forward/backward
                try:
                    global_step = pbar.n + 1
                    if global_step > pbar.total or continual_error >= MAX_CONTINUAL_ERROR:
                        if continual_error >= MAX_CONTINUAL_ERROR:
                            log.error(f"Reach max continual error {MAX_CONTINUAL_ERROR} due OOM or NaN gradient. Please "
                                       "reduce the batch size / learning rate, or re-check the model's numerical stability")
                            exit(1)
                        else:
                            log.info("The training successfully completes")

                        if is_leader_process():
                            log.info("Saving the final states")
                            save_states(global_step, epoch, [f"states-{global_step - 1}.ckpt"])
                        training_completed = True
                        break

                    if backward_steps == 0:
                        find_tensor = False
                        for item in [wavs, *others]:
                            if isinstance(item, (list, dict)):
                                for sub_item in item:
                                    if isinstance(sub_item, torch.Tensor):
                                        find_tensor = True
                            elif isinstance(item, torch.Tensor):
                                find_tensor = True
                        if find_tensor:
                            log.warning("We do not recommend to return torch.Tensor from the dataloader "
                                        "since it can cause out-of-shared-memory when torch's multiprocess "
                                        "sharing strategy is set to file_system. Please consider to return "
                                        "numpy in your dataloader. If you are using the existing downstream "
                                        "and get this message. Please help open an issue. Thanks!")

                    wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                    if self.upstream.trainable:
                        features = self.upstream(wavs)
                    else:
                        with torch.no_grad():
                            features = self.upstream(wavs)
                    features = self.featurizer(wavs, features)

                    if specaug:
                        features, _ = specaug(features)

                    loss = self.downstream(
                        train_split,
                        features, *others,
                        records = records,
                    )
                    batch_ids.append(batch_id)

                    (loss / gradient_accumulate_steps).backward()
                    del loss

                except RuntimeError as e:
                    acceptable = False
                    for acc_err in ACCEPTABLE_ERRORS:
                        if acc_err in str(e):
                            acceptable = True

                    if acceptable:
                        if is_initialized():
                            raise
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continual_error += 1
                        log.warning(f'Error counter {continual_error}: CUDA out of memory at step {global_step}')
                        continue
                    else:
                        raise

                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_paras, self.config['runner']['gradient_clipping'])

                # optimize
                if math.isnan(grad_norm):
                    optimizer.zero_grad()
                    continual_error += 1
                    log.warning(f'Error counter {continual_error}: Grad norm is NaN at step {global_step}')
                    continue
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()

                # This optimization successfully completes
                continual_error = 0
                pbar.update(1)

                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    if is_initialized(): torch.distributed.barrier()
                    continue

                # logging
                if global_step % self.config['runner']['log_step'] == 0:
                    self.downstream.log_records(
                        train_split,
                        records = records,
                        logger = logger,
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
                        save_names += self.evaluate(split, logger, global_step)

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
                    save_states(global_step, epoch, save_names)

                if is_initialized(): torch.distributed.barrier()

        pbar.close()
        if is_leader_process():
            logger.close()
            if self.args.push_to_hf_hub:
                self.push_to_huggingface_hub()


    @torch.no_grad()
    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # prepare data
        dataloader = self.downstream.get_dataloader(split)
        total_steps = len(dataloader)
        eval_batch = self.config["runner"].get("eval_batch")
        if eval_batch is not None:
            assert isinstance(eval_batch, int)
            total_steps = eval_batch

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=total_steps)):
            if batch_id > total_steps:
                break

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            features = self.upstream(wavs)
            features = self.featurizer(wavs, features)
            self.downstream(
                split,
                features, *others,
                records = records,
                batch_id = batch_id,
            )
            batch_ids.append(batch_id)

        save_names = self.downstream.log_records(
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
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        return [] if type(save_names) is not list else save_names

    def inference(self):
        filepath = Path(self.args.evaluate_split)
        assert filepath.is_file(), filepath
        filename = filepath.stem

        if hasattr(self.downstream.model, "load_audio"):
            wav = self.downstream.model.load_audio(filepath)
        else:
            wav, sr = torchaudio.load(str(filepath))
            assert sr == SAMPLE_RATE, sr
        wavs = [wav.view(-1).to(self.args.device)]

        for entry in self.all_entries:
            entry.model.eval()

        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            self.downstream.model.inference(features, [filename])

    def push_to_huggingface_hub(self):
        """Creates a downstream repository on the Hub and pushes training artifacts to it."""
        if self.args.hf_hub_org.lower() != "none":
            organization = self.args.hf_hub_org
        else:
            organization = os.environ.get("HF_USERNAME")
        huggingface_token = HfFolder.get_token()
        log.info(f"Organisation to push fine-tuned model to: {organization}")

        # Extract upstream repository metadata
        if self.args.hub == "huggingface":
            model_info = HfApi().model_info(self.args.upstream, token=huggingface_token)
            downstream_model_id = model_info.sha
            # Exclude "/" characters from downstream repo ID
            upstream_model_id = model_info.modelId.replace("/", "__")
        else:
            upstream_model_id = self.args.upstream.replace("/", "__")
            downstream_model_id = str(uuid.uuid4())[:8]
        repo_name = f"{upstream_model_id}__{downstream_model_id}"
        # Create downstream repo on the Hub
        repo_url = HfApi().create_repo(
            token=huggingface_token,
            name=repo_name,
            organization=organization,
            exist_ok=True,
            private=False,
        )
        log.info(f"Created Hub repo: {repo_url}")

        # Download repo
        HF_HUB_DIR = "hf_hub"
        REPO_ROOT_DIR = os.path.join(self.args.expdir, HF_HUB_DIR, repo_name)
        REPO_TASK_DIR = os.path.join(REPO_ROOT_DIR, self.args.downstream, self.args.expname)
        log.info(f"Cloning Hub repo to {REPO_ROOT_DIR}")
        model_repo = Repository(
            local_dir=REPO_ROOT_DIR, clone_from=repo_url, use_auth_token=huggingface_token
        )
        # Pull latest changes if they exist
        model_repo.git_pull()

        # Copy checkpoints, tensorboard logs, and args / configs
        # Note that this copies all files from the experiment directory,
        # including those from multiple runs
        shutil.copytree(self.args.expdir, REPO_TASK_DIR, dirs_exist_ok=True, ignore=shutil.ignore_patterns(HF_HUB_DIR))

        # By default we use model.ckpt in the PreTrainedModel interface, so
        # rename the best checkpoint to match this convention
        checkpoints = list(Path(REPO_TASK_DIR).glob("*best*.ckpt"))
        if len(checkpoints) == 0:
            log.warning("Did not find a best checkpoint! Using the final checkpoint instead ...")
            CKPT_PATH = (
                os.path.join(REPO_TASK_DIR, f"states-{self.config['runner']['total_steps']}.ckpt")
                )
        elif len(checkpoints) > 1:
            log.warning(f"More than one best checkpoint found! Using {checkpoints[0]} as default ...")
            CKPT_PATH = checkpoints[0]
        else:
            log.info(f"Found best checkpoint {checkpoints[0]}!")
            CKPT_PATH = checkpoints[0]
        shutil.move(CKPT_PATH, os.path.join(REPO_TASK_DIR, "model.ckpt"))
        model_repo.lfs_track("*.ckpt")

        # Write model card
        self._create_model_card(REPO_ROOT_DIR)

        # Push everything to the Hub
        log.info("Pushing model files to the Hub ...")
        model_repo.push_to_hub()
        log.info("Training run complete!")
