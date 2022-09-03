import argparse
import inspect
import logging
import math
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import time
from typing import List

import omegaconf
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from s3prl.nn.interface import AbsFeaturizer, AbsUpstream
from s3prl.nn.upstream import Featurizer, S3PRLUpstream, UpstreamDownstreamModel
from s3prl.sampler import DistributedBatchSamplerWrapper
from s3prl.util.override import parse_overrides
from s3prl.util.seed import fix_random_seeds

logger = logging.getLogger(__name__)

LOGGING_FORMAT = "%(levelname)s | %(asctime)s | %(module)s:%(lineno)d | %(message)s"
ACCEPTABLE_ERRORS = [
    "CUDA out of memory",
    "Unable to find a valid cuDNN algorithm to run convolution",  # Usually caused by CUDA OOM
]


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def force_cacheable(data: dict):
    output = dict()
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        output[key] = value
    return output


def to_device(data, device: str):
    output = dict()
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
        output[key] = value
    return output


class Utility:
    @classmethod
    def build_collate_fn(cls, _mode: str):
        from s3prl.dataset.base import default_collate_fn

        return default_collate_fn

    @classmethod
    def build_upstream(cls, name: str) -> AbsUpstream:
        """
        From waveform to a list of hidden states
        """
        upstream = S3PRLUpstream(name)
        return upstream

    @classmethod
    def build_featurizer(
        cls, _upstream, layer_selections: List[int], normalize: bool
    ) -> AbsFeaturizer:
        """
        Reduce a list of hidden states to a single hidden state
        """
        featurizer = Featurizer(
            _upstream, layer_selections=layer_selections, normalize=normalize
        )
        return featurizer

    @classmethod
    def build_model(
        cls,
        _model_output_size: str,
        _build_upstream: dict,
        _build_featurizer: dict,
        _build_downstream: dict,
        upstream_trainable: bool,
    ):
        upstream = cls.build_upstream(**_build_upstream)
        featurizer: Featurizer = cls.build_featurizer(upstream, **_build_featurizer)
        downstream = cls.build_downstream(
            featurizer.output_size,
            _model_output_size,
            featurizer.downsample_rate,
            **_build_downstream,
        )
        model = UpstreamDownstreamModel(
            upstream, featurizer, downstream, upstream_trainable
        )
        return model

    @classmethod
    def build_optimizer(cls, _parameters, name: str, conf: dict):
        opt_cls = getattr(torch.optim, name)
        opt = opt_cls(_parameters, **conf)
        return opt

    @classmethod
    def build_scheduler(cls, _optimizer, name: str, conf: dict):
        scheduler_cls = getattr(torch.optim.lr_scheduler, name)
        scheduler = scheduler_cls(_optimizer, **conf)
        return scheduler

    @classmethod
    def _get_current_arguments(cls) -> dict:
        frame = inspect.currentframe().f_back
        args, _, _, values = inspect.getargvalues(frame)

        config = dict()
        for key in args[1:]:
            config[key] = values[key]

        return config

    @classmethod
    def _get_time_tag(cls):
        return datetime.fromtimestamp(time()).strftime("%Y_%m_%d_%H_%M_%S")

    @classmethod
    def _stage_check(cls, stage_id: int, stop: int, check_fn: callable):
        try:
            check_fn()
        except:
            logger.error(
                f"Stage {stage_id} was not done before or is corrupted. Please re-run from this stage."
            )
            raise
        if isinstance(stop, int) and stage_id == stop:
            exit(0)

    @classmethod
    def train(
        cls,
        _train_dir: str,
        _init_model: dict,
        _init_task: dict,
        _save_model: dict,
        _save_task: dict,
        _build_optimizer: dict,
        _build_scheduler: dict,
        _train_dataset,
        _train_batch_sampler,
        _valid_dataset,
        _valid_batch_sampler,
        _num_workers: int,
        _world_size: int,
        _rank: int,
        _eval_batch: int,
        _device: str,
        total_steps: int,
        log_step: int,
        eval_step: int,
        save_step: int,
        gradient_clipping: float,
        gradient_accumulate_steps: int,
        valid_metric: str,
        valid_higher_better: bool,
        auto_resume: bool = True,
        resume_ckpt_dir: str = None,
        seed: int = 0,
        keep_num_ckpts: int = 2,
        use_scheduler: bool = False,
    ):
        fix_random_seeds(seed)

        _train_dir: Path = Path(_train_dir)
        if not auto_resume and _train_dir.is_dir():
            logger.warning(
                f"{_train_dir} exists. Delete the directory since auto_resume=False"
            )
            shutil.rmtree(_train_dir)
        _train_dir.mkdir(exist_ok=True, parents=True)

        ckpt_dirs = [key for key in os.listdir(_train_dir) if key.startswith("step_")]
        ckpt_dirs.sort(key=lambda name: int(name.split("_")[-1]), reverse=True)

        resume = False
        if auto_resume:
            if resume_ckpt_dir is not None and Path(resume_ckpt_dir).is_dir():
                resume = True
            if len(ckpt_dirs) > 0:
                resume = True

        if resume:
            resume_ckpt_dir = Path(resume_ckpt_dir or _train_dir / ckpt_dirs[0])
            logger.info(f"Loading checkpoints from {resume_ckpt_dir}")
            try:
                _, _, task, _ = cls.load_model_and_task(resume_ckpt_dir)
            except:
                logger.error(
                    "Fail to load the checkpoint. "
                    "The config-specified task or model is not the same one as that saved in the checkpoint. "
                    "You can set '--train.auto_resume False' to ignore the old checkpoint to avoid this behavior."
                )
                raise

            optimizer_state = torch.load(
                resume_ckpt_dir / "optimizer.pt", map_location="cpu"
            )

            if use_scheduler:
                scheduler_state = torch.load(
                    resume_ckpt_dir / "scheduler.pt", map_location="cpu"
                )
            else:
                scheduler_state = None

            with open(resume_ckpt_dir / "training_stats.yaml", "r") as f:
                training_stats = yaml.load(f, Loader=yaml.FullLoader)

            global_step = int(training_stats["global_step"])
            epoch = int(training_stats["epoch"])
            valid_best_metrics = dict(training_stats["valid_best_metrics"])

        else:
            model = cls.build_model(**_init_model)
            task = cls.build_task(model, **_init_task)
            optimizer_state = None
            scheduler_state = None
            global_step = 0
            epoch = 0
            valid_best_metrics = dict()

        device = torch.device(_device)
        wrapped_task = task.to(device)

        if _world_size > 1:
            torch.cuda.set_device(device.index)
            wrapped_task = DistributedDataParallel(
                task,
                device_ids=[device.index],
                find_unused_parameters=True,
                output_device=device.index,
            )

        optimizer = cls.build_optimizer(task.parameters(), **_build_optimizer)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        scheduler = None
        if use_scheduler:
            scheduler = cls.build_scheduler(optimizer, **_build_scheduler)
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)

        _train_batch_sampler = DistributedBatchSamplerWrapper(
            _train_batch_sampler,
            num_replicas=_world_size,
            rank=_rank,
        )
        train_dataloader = DataLoader(
            _train_dataset,
            batch_sampler=_train_batch_sampler,
            num_workers=_num_workers,
            collate_fn=cls.build_collate_fn("train"),
        )

        tqdm_file = sys.stderr if _rank == 0 else open(os.devnull, "w")
        pbar = tqdm(
            total=total_steps,
            dynamic_ncols=True,
            desc="train",
            file=tqdm_file,
        )
        pbar.n = global_step

        if _rank == 0:
            tf_dir = _train_dir / "tb"
            tf_logger = SummaryWriter(str(tf_dir))

        backward_steps = 0
        while pbar.n < pbar.total:
            _train_batch_sampler.set_epoch(epoch),
            batch_results = []
            logger.info(f"Start epoch {epoch}")
            for batch in train_dataloader:
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    wrapped_task.train()
                    batch = to_device(batch, device)
                    loss, cacheable = wrapped_task("train", **batch)
                    (loss / gradient_accumulate_steps).backward()
                    batch_results.append(force_cacheable(cacheable))

                except RuntimeError as e:
                    if _world_size > 1:
                        raise

                    acceptable = False
                    for acc_err in ACCEPTABLE_ERRORS:
                        if str(e) in acc_err:
                            acceptable = True
                            break
                    if not acceptable:
                        raise

                    logger.warning(f"Step {global_step}: {str(e)}")
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue

                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    wrapped_task.parameters(), gradient_clipping
                )

                if math.isnan(grad_norm):
                    logger.warning(f"[Runner] - grad norm is NaN at step {global_step}")
                else:
                    optimizer.step()
                optimizer.zero_grad()

                if use_scheduler:
                    scheduler.step()

                if _rank > 0:
                    batch_results = []
                    pbar.update(1)
                    continue

                if global_step % log_step == 0:
                    logs = wrapped_task.reduction("train", batch_results)
                    cls._log_results("train", logs, tf_logger, global_step)
                    batch_results = []

                save_names = []

                if global_step % eval_step == 0:
                    logs: dict = cls._evaluate(
                        "valid",
                        task,
                        _valid_dataset,
                        _valid_batch_sampler,
                        _eval_batch,
                        _train_dir,
                        _device,
                        _num_workers,
                    )
                    cls._log_results("valid", logs, tf_logger, global_step)
                    valid_metrics = {k: float(v) for k, v in logs.items()}
                    new_metric = valid_metrics[valid_metric]
                    best_metric = valid_best_metrics.get(valid_metric)
                    if best_metric is None:
                        is_new_best = True
                    elif valid_higher_better:
                        is_new_best = new_metric > best_metric
                    else:
                        is_new_best = new_metric < best_metric
                    if is_new_best:
                        valid_best_metrics = deepcopy(valid_metrics)
                        save_names.append("valid_best")

                if global_step % save_step == 0:
                    ckpt_dirs = [
                        key for key in os.listdir(_train_dir) if key.startswith("step_")
                    ]
                    ckpt_dirs.sort(key=lambda stem: int(stem.split("_")[-1]))
                    if len(ckpt_dirs) >= keep_num_ckpts:
                        for ckpt_dir in ckpt_dirs[
                            : len(ckpt_dirs) - keep_num_ckpts + 1
                        ]:
                            shutil.rmtree(_train_dir / ckpt_dir)

                    save_names.append(f"step_{global_step}")

                for name in save_names:
                    training_stats = dict(
                        global_step=global_step,
                        epoch=epoch,
                        valid_best_metrics=valid_best_metrics,
                    )
                    cls._save_ckpts_to_dir(
                        _train_dir / name,
                        (
                            task.module
                            if isinstance(task, DistributedDataParallel)
                            else task
                        ),
                        optimizer,
                        scheduler,
                        _init_model,
                        _save_model,
                        _init_task,
                        _save_task,
                        training_stats,
                    )

                pbar.update(1)
            epoch += 1

        pbar.close()
        if _rank == 0:
            tf_logger.close()

    @classmethod
    def _evaluate(
        cls,
        _mode: str,
        _task,
        _dataset,
        _batch_sampler,
        _eval_batch: int,
        _dump_dir: str,
        _device: str,
        _num_workers: int,
    ):
        assert _mode in ["valid", "test"]

        dataloader = DataLoader(
            _dataset,
            batch_sampler=_batch_sampler,
            num_workers=_num_workers,
            collate_fn=cls.build_collate_fn(_mode),
        )

        task = _task.to(_device)
        with torch.no_grad():
            batch_results = []
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc=_mode, total=len(dataloader))
            ):
                if batch_idx == _eval_batch:
                    break
                batch = to_device(batch, _device)
                task.eval()
                loss, cacheable = task(_mode, _dump_dir=_dump_dir, **batch)
                batch_results.append(force_cacheable(cacheable))

        logs = task.reduction(_mode, batch_results, _dump_dir=_dump_dir)
        return logs

    @classmethod
    def _log_results(
        cls,
        split_name: str,
        logs: dict,
        tensorboard: SummaryWriter,
        global_step: int,
    ):
        logger.info(f"{split_name} at step {global_step}")
        for name, value in logs.items():
            value = float(value)
            logger.info(f"{name}: {value}")
            tensorboard.add_scalar(
                f"{split_name}-{name}", value, global_step=global_step
            )

    @classmethod
    def _save_yaml(cls, data, path):
        Path(path).parent.mkdir(exist_ok=True, parents=True)

        with Path(path).open("w") as f:
            yaml.dump(data, f)

        try:
            with open(path, "r") as f:
                yaml.load(f, Loader=yaml.FullLoader)
        except:
            logger.error(
                f"The following data can not be safely serialized to yaml: {data}"
            )
            raise

    @classmethod
    def _save_ckpts_to_dir(
        cls,
        _ckpts_dir: str,
        _task,
        _optimizer,
        _scheduler,
        _init_model: dict,
        _save_model: dict,
        _init_task: dict,
        _save_task: dict,
        _training_stats: dict,
    ):
        ckpts_dir: Path = Path(_ckpts_dir)
        ckpts_dir.mkdir(exist_ok=True, parents=True)

        model_ckpt_path = ckpts_dir / "model.pt"
        cls.save_model(model_ckpt_path, _init_model, _task.model, **_save_model)

        task_ckpt_path = ckpts_dir / "task.pt"
        cls.save_task(task_ckpt_path, _init_task, _task, **_save_task)

        torch.save(_optimizer.state_dict(), ckpts_dir / "optimizer.pt")
        if _scheduler is not None:
            torch.save(_scheduler.state_dict(), ckpts_dir / "scheduler.pt")

        cls._save_yaml(_training_stats, _ckpts_dir / "training_stats.yaml")

    @classmethod
    def save_model(
        cls,
        _model_ckpt_path: str,
        _init_model: dict,
        _model: torch.nn.Module,
        extra_conf: dict = None,
    ):
        state = {
            "init_model": _init_model,
            "model_weight": _model.state_dict(),
            "extra_conf": extra_conf or dict(),
        }
        Path(_model_ckpt_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(state, _model_ckpt_path)

    @classmethod
    def load_model(cls, _model_ckpt_path: str):
        ckpt = torch.load(_model_ckpt_path, map_location="cpu")
        model: torch.nn.Module = cls.build_model(**ckpt["init_model"])
        model.load_state_dict(ckpt["model_weight"])

        model_extra_conf = ckpt["extra_conf"]
        return model, model_extra_conf

    @classmethod
    def save_task(
        cls,
        _task_ckpt_path: str,
        _init_task: dict,
        _task,
        extra_conf: dict = None,
    ):
        state = {
            "init_task": _init_task,
            "task_state": _task.get_state(),
            "extra_conf": extra_conf,
        }
        Path(_task_ckpt_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(state, _task_ckpt_path)

    @classmethod
    def load_model_and_task(cls, _ckpts_dir: str):
        model_ckpt_path = Path(_ckpts_dir) / "model.pt"
        model, model_extra_conf = cls.load_model(model_ckpt_path)

        task_ckpt_path = Path(_ckpts_dir) / "task.pt"
        ckpt = torch.load(task_ckpt_path, map_location="cpu")
        task = cls.build_task(model, **ckpt["init_task"])
        task.set_state(ckpt["task_state"])

        task_extra_conf = ckpt["extra_conf"]
        return model, model_extra_conf, task, task_extra_conf

    @classmethod
    def main(cls, args: List[str] = None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", default="INFO")
        parser.add_argument(
            "--config", help="The yaml config path to override the default config"
        )
        parser.add_argument("--print_config", "-p", action="store_true")
        parser.add_argument(
            "--dump_config", "-d", help="The path to dump the default config as yaml"
        )
        args, override = parser.parse_known_args(args)

        if args.print_config:
            print(f"\nDefault config of {cls}\n")
            print(yaml.dump(cls.default_config()))
            exit(0)

        if args.dump_config is not None:
            with open(args.dump_config, "w") as f:
                yaml.dump(cls.default_config(), f)
            exit(0)

        root_logger = logging.getLogger()
        root_logger.handlers = []
        logging.basicConfig(level=getattr(logging, args.verbose), format=LOGGING_FORMAT)

        if args.config is not None:
            with open(args.config) as f:
                yaml_conf = yaml.load(f, Loader=yaml.FullLoader) or dict()
        else:
            yaml_conf = dict()
        override_conf = parse_overrides(override)

        schema = omegaconf.OmegaConf.create(cls.default_config())
        config = omegaconf.OmegaConf.merge(schema, yaml_conf, override_conf)
        config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        logger.info(config)

        cls.run(**config)
        return config
