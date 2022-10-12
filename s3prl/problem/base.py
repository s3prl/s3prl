"""
The shared backbone of common ML train/test procedure for all problems

Authors:
  * Leo 2022
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import os
import pickle
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Union

import omegaconf
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from s3prl.dataio.sampler import DistributedBatchSamplerWrapper
from s3prl.dataset.base import default_collate_fn
from s3prl.nn.upstream import Featurizer, S3PRLUpstream, UpstreamDownstreamModel
from s3prl.task import Task
from s3prl.util.override import parse_overrides
from s3prl.util.seed import fix_random_seeds

logger = logging.getLogger(__name__)

LOGGING_FORMAT = "%(levelname)s | %(asctime)s | %(module)s:%(lineno)d | %(message)s"
ACCEPTABLE_ERRORS = [
    "CUDA out of memory",
    "Unable to find a valid cuDNN algorithm to run convolution",  # Usually caused by CUDA OOM
]
PRIMITIVE_TYPES = (int, float, bool, str)

DEFAULT_CONFIG_FORMAT = """
The default arguments for :obj:`run` in yaml.
Note that for the fields with inner values, like :code:`build_model`,
the outer field name corresponds to a method name, so you can find the method
:obj:`build_model`. Furthermore, the values inside that field will be
directly passed into the method. So by changing these inner values, you
can directly affect the behavior of the corresponding method. See the method
documentation for all the supported arguments and their meanings.

The methods affected by the following config are: {:s}

.. code-block:: yaml

{:s}
"""

__all__ = ["Problem"]


class _DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def _force_cacheable(data: dict):
    output = dict()
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        output[key] = value
    return output


def _to_device(data, device: str):
    output = dict()
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
        output[key] = value
    return output


def _doc_default_config(cls: Problem):
    """
    This is used to layout the :code:`default_config` dictionary into yaml format
    for :code:`default_config`'s docstring.
    """

    def _append_prefix_spaces(docstring: str):
        return "\n".join([f"  {line}" for line in docstring.split("\n")])

    obj = cls()
    try:
        config = obj.default_config()
    except:
        return
    else:
        methods = []
        for k, v in config.items():
            if hasattr(cls, k):
                methods.append(getattr(cls, k))
        method_links = " ".join([f":obj:`{method.__name__}`" for method in methods])

        yaml_str = yaml.dump(config, sort_keys=False, width=float("inf"))
        yaml_str = _append_prefix_spaces(yaml_str)
        cls.default_config.__doc__ = DEFAULT_CONFIG_FORMAT.format(
            method_links, yaml_str
        )


class Problem:
    _store: Dict[str, Problem] = dict()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._store[cls.__name__] = cls
        _doc_default_config(cls)

    @classmethod
    def get_class_from_name(cls, name: str):
        """
        Args:
            name (str): the :code:`__name__` of the problem class

        Returns:
            Problem
        """
        assert (
            name in cls._store
        ), f"The class '{name}' is either not defined or not imported"
        return cls._store[name]

    def build_collate_fn(self, build_collate_fn: dict, mode: str):
        """
        By default returns :obj:`s3prl.dataset.base.default_collate_fn`

        Args:
            build_collate_fn (dict): same in :obj:`default_config`, no argument supported for now
            mode (str): train, valid, or test

        Returns:
            callable

            the collate_fn for torch DataLoader in train/valid/test :code:`mode`
        """
        return default_collate_fn

    def build_upstream(self, build_upstream: dict):
        """
        By default build the upstream with :obj:`s3prl.nn.upstream.S3PRLUpstream`

        Args:
            build_upstream (dict): same in :obj:`default_config`,
                arguments for :obj:`s3prl.nn.upstream.S3PRLUpstream`

        Returns:
            :obj:`s3prl.nn.interface.AbsUpstream`

            Return an upstream model, whose forward takes the waveform input and returns
            multiple hidden states as features.
        """
        upstream = S3PRLUpstream(**build_upstream)
        return upstream

    def build_featurizer(self, build_featurizer: dict, upstream):
        """
        By default build the featurizer with :obj:`s3prl.nn.Featurizer`

        Args:
            build_featurizer (dict): same in :obj:`default_config`,
                arguments for :obj:`s3prl.nn.Featurizer`
            upstream (:obj:`AbsUpstream`): the upstream model built by :obj:`build_upstream`

        Returns:
            :obj:`s3prl.nn.interface.AbsFeaturizer`

            Return the featurizer model. The featurizer is used to reduce the multiple
            hidden states returned from the upstream model (built by :obj:`build_upstream`)
            into a single hidden state, so can be easliy fed into the downstream model
        """
        featurizer = Featurizer(upstream, **build_featurizer)
        return featurizer

    def build_model(
        self,
        build_model: dict,
        model_output_size: int,
        build_upstream: dict,
        build_featurizer: dict,
        build_downstream: dict,
    ):
        """
        By default build model with :obj:`s3prl.nn.upstream.UpstreamDownstreamModel`

        Args:
            build_model (dict): same in :obj:`default_config`,
                arguments for :obj:`s3prl.nn.upstream.UpstreamDownstreamModel`
            model_output_size (int): the required model's output hidden size
            build_upstream (dict): same in :obj:`default_config`, refer to :obj:`build_upstream`
            build_featurizer (dict): same in :obj:`default_config`, refer to :obj:`build_featurizer`
            build_downstream (dict): same in :obj:`default_config`, refer to :obj:`build_downstream`

        Returns:
            torch.nn.Module

            Return the entire model for the task, which takes the direct items from DataLoader as the input.
            Usually, the components can be built by :obj:`build_upstream`, :obj:`build_featurizer`,
            :obj:`build_downstream`, and are concated together to get the final model.
            The upstream extracts multiple hidden states, the featuizer reduce them into a single hidden state,
            and the downstream takes the hidden states as the feature for the downstream-specific model.
        """
        upstream = self.build_upstream(build_upstream)
        featurizer: Featurizer = self.build_featurizer(build_featurizer, upstream)
        downstream = self.build_downstream(
            build_downstream,
            featurizer.output_size,
            model_output_size,
            featurizer.downsample_rate,
        )
        model = UpstreamDownstreamModel(upstream, featurizer, downstream, **build_model)
        return model

    def build_optimizer(self, build_optimizer: dict, parameters):
        """
        Args:
            build_optimizer (dict): same in :obj:`default_config`, refer to below

                ====================  ====================
                key                   description
                ====================  ====================
                name                  (str) - the optimizer class name in :obj:`torch.optim`
                conf                  (dict) - the arguments for initializing the optimizer class. e.g. :code:`{"lr": 1.0e-4}`
                ====================  ====================

            parameters (iterable): the standard params accepted by :obj:`torch.optim.Optimizer`.

        Returns:
            :obj:`torch.optim.Optimizer`

            An optimizer following standard torch usage
        """

        def _default_build_optimizer(name: str, conf: dict):
            opt_cls = getattr(torch.optim, name)
            opt = opt_cls(parameters, **conf)
            return opt

        return _default_build_optimizer(**build_optimizer)

    def build_scheduler(self, build_scheduler: dict, optimizer):
        """
        Args:
            build_scheduler (dict): same in :obj:`default_config`

                ====================  ====================
                key                   description
                ====================  ====================
                name                  (str) - the scheduler class name in :obj:`torch.optim.lr_scheduler`
                conf                  (dict) - the arguments for initializing the scheduler class. e.g. :code:`{"gamma": 0.01}` for :obj:`torch.optim.lr_scheduler.StepLR`
                ====================  ====================

            optimizer: the standard torch optimizer accepted by Scheduler in :obj:`torch.optim.lr_scheduler`.

        Returns:
            torch scheduler

            A scheduler following standard torch usage
        """

        def _default_build_scheduler(name: str, conf: dict):
            scheduler_cls = getattr(torch.optim.lr_scheduler, name)
            scheduler = scheduler_cls(optimizer, **conf)
            return scheduler

        return _default_build_scheduler(**build_scheduler)

    def train(
        self,
        train: dict,
        train_dir: str,
        build_model_all_args: dict,
        build_task_all_args_except_model: dict,
        save_model: dict,
        save_task: dict,
        build_optimizer: dict,
        build_scheduler: dict,
        evaluate: dict,
        train_dataset,
        train_batch_sampler,
        train_collate_fn,
        valid_dataset,
        valid_batch_sampler,
        valid_collate_fn,
        num_workers: int,
        world_size: int,
        rank: int,
        eval_batch: int,
        device: str,
        global_config: dict = None,
    ):
        """
        Args:
            train (dict): same in :obj:`default_config`

                ==========================  ====================
                key                         description
                ==========================  ====================
                total_steps                 (int) - the total optimization steps
                log_step                    (int) - logging frequency. log every :code:`log_step` step
                eval_step                   (int) - evaluation frequency. Evaluate every :code:`eval_step` step. \
                                                Note that you can control how many batch to evaluate to speed up the \
                                                development by the :code:`eval_batch` argument in :obj:`run`
                save_step                   (int) - save the checkpoint every :code:`save_step` step.
                gradient_clipping           (float) - clip the gradient. important for RNNs.
                gradient_accumulate         (int) - accumulate multiple steps' gradient before updating network parameters \
                                                to simulate large-batch optimization.
                valid_metric                (str) - the metric to select the best valid checkpoint. Different Tasks have different \
                                                supported valid_metrics. See :obj:`build_task` for the supported metrics.
                valid_higher_better         (bool) - some metrics are higher better, while some are lower better \
                                                this will affect how to save the best validation checkpoint.
                auto_resume                 (bool) - if there are already the last checkpoint in :code:`target_dir` (see :obj:`run`), \
                                                whether to resume from it or delete it and start a new training session.
                resume_ckpt_dir             (str) - you can directly specify the checkpoint path to resume which is not necessary \
                                                in :code:`target_dir` (see :obj:`run`).
                seed                        (int) - fix the seed before the training start
                keep_num_ckpts              (int) - to prevent saving too many checkpoints, only save the :code:`keep_num_ckpts` \
                                                latest checkpoints and delete the old ones.
                use_scheduler               (bool) - whether to use the scheduler
                ==========================  ====================

            **others:
                only meaningful when you want to override this train method, which is not the
                common case. Hence we skip the documentation for now.
        """

        @dataclass
        class TrainConfig:
            total_steps: int
            log_step: int
            eval_step: int
            save_step: int
            gradient_clipping: float
            gradient_accumulate: int
            valid_metric: str
            valid_higher_better: bool
            auto_resume: bool = True
            resume_ckpt_dir: str = None
            seed: int = 0
            keep_num_ckpts: int = 2
            use_scheduler: bool = False

        conf = TrainConfig(**train)

        fix_random_seeds(conf.seed)

        train_dir: Path = Path(train_dir)
        if not conf.auto_resume and train_dir.is_dir():
            logger.warning(
                f"{train_dir} exists. Delete the directory since auto_resume=False"
            )
            shutil.rmtree(train_dir)
        train_dir.mkdir(exist_ok=True, parents=True)

        ckpt_dirs = [key for key in os.listdir(train_dir) if key.startswith("step_")]
        ckpt_dirs.sort(key=lambda name: int(name.split("_")[-1]), reverse=True)

        resume = False
        if conf.auto_resume:
            if conf.resume_ckpt_dir is not None and Path(conf.resume_ckpt_dir).is_dir():
                resume = True
            if len(ckpt_dirs) > 0:
                resume = True

        if resume:
            resume_ckpt_dir = Path(conf.resume_ckpt_dir or train_dir / ckpt_dirs[0])
            logger.info(f"Loading checkpoints from {resume_ckpt_dir}")
            try:
                _, task = self.load_model_and_task(resume_ckpt_dir)
            except:
                logger.error(
                    f"Fail to load the checkpoint {resume_ckpt_dir}. "
                    "You can set '--train.auto_resume False' to ignore the crashed checkpoint to avoid this behavior."
                )
                raise

            optimizer_state = torch.load(
                resume_ckpt_dir / "optimizer.pt", map_location="cpu"
            )

            if conf.use_scheduler:
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
            model = self.build_model(**build_model_all_args)
            task = self.build_task(model=model, **build_task_all_args_except_model)
            optimizer_state = None
            scheduler_state = None
            global_step = 0
            epoch = 0
            valid_best_metrics = dict()

        device = torch.device(device)
        wrapped_task = task.to(device)

        if world_size > 1:
            torch.cuda.set_device(device.index)
            wrapped_task = _DistributedDataParallel(
                task,
                device_ids=[device.index],
                find_unused_parameters=True,
                output_device=device.index,
            )

        optimizer = self.build_optimizer(build_optimizer, task.parameters())
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        scheduler = None
        if conf.use_scheduler:
            scheduler = self.build_scheduler(build_scheduler, optimizer)
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)

        train_batch_sampler = DistributedBatchSamplerWrapper(
            train_batch_sampler,
            num_replicas=world_size,
            rank=rank,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
        )

        tqdm_file = sys.stderr if rank == 0 else open(os.devnull, "w")
        pbar = tqdm(
            total=conf.total_steps,
            dynamic_ncols=True,
            desc="train",
            file=tqdm_file,
        )
        pbar.n = global_step

        if rank == 0:
            tf_dir = train_dir / "tb"
            tf_logger = SummaryWriter(str(tf_dir))

        def _save_ckpts_to_dir(
            ckpts_dir: str,
            task,
            optimizer,
            scheduler,
            build_model_all_args: dict,
            build_task_all_args_except_model: dict,
            save_model: dict,
            save_task: dict,
            training_stats: dict,
            global_config: dict,
        ):
            ckpts_dir: Path = Path(ckpts_dir)
            ckpts_dir.mkdir(exist_ok=True, parents=True)

            model_ckpt_dir = ckpts_dir / "model"
            self.save_model(
                save_model, model_ckpt_dir, build_model_all_args, task.model
            )

            task_ckpt_dir = ckpts_dir / "task"
            self.save_task(
                save_task, task_ckpt_dir, build_task_all_args_except_model, task
            )

            torch.save(optimizer.state_dict(), ckpts_dir / "optimizer.pt")
            if scheduler is not None:
                torch.save(scheduler.state_dict(), ckpts_dir / "scheduler.pt")

            with (ckpts_dir / "training_stats.yaml").open("w") as f:
                yaml.safe_dump(training_stats, f)

            with (ckpts_dir / "config.yaml").open("w") as f:
                yaml.safe_dump(global_config, f)

        backward_steps = 0
        while pbar.n < pbar.total:
            train_batch_sampler.set_epoch(epoch),
            batch_results = []
            logger.info(f"Start epoch {epoch}")
            for batch in train_dataloader:
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    wrapped_task.train()
                    batch = _to_device(batch, device)
                    loss, cacheable = wrapped_task("train", **batch)
                    (loss / conf.gradient_accumulate).backward()
                    batch_results.append(_force_cacheable(cacheable))

                except RuntimeError as e:
                    if world_size > 1:
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
                if backward_steps % conf.gradient_accumulate > 0:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    wrapped_task.parameters(), conf.gradient_clipping
                )

                if math.isnan(grad_norm):
                    logger.warning(f"[Runner] - grad norm is NaN at step {global_step}")
                else:
                    optimizer.step()
                optimizer.zero_grad()

                if conf.use_scheduler:
                    scheduler.step()

                if rank > 0:
                    batch_results = []
                    pbar.update(1)
                    continue

                def _log_results(
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

                if global_step % conf.log_step == 0:
                    logs = wrapped_task.reduction("train", batch_results)
                    _log_results("train", logs, tf_logger, global_step)
                    batch_results = []

                save_names = []

                if global_step % conf.eval_step == 0:
                    assert (
                        valid_dataset is not None and valid_batch_sampler is not None
                    ), f"valid dataset is not supported, please set train.eval_step to infinite"
                    logs: dict = self.evaluate(
                        evaluate,
                        "valid",
                        task,
                        valid_dataset,
                        valid_batch_sampler,
                        valid_collate_fn,
                        eval_batch,
                        train_dir,
                        device,
                        num_workers,
                    )
                    _log_results("valid", logs, tf_logger, global_step)
                    valid_metrics = {k: float(v) for k, v in logs.items()}
                    new_metric = valid_metrics[conf.valid_metric]
                    best_metric = valid_best_metrics.get(conf.valid_metric)
                    if best_metric is None:
                        is_new_best = True
                    elif conf.valid_higher_better:
                        is_new_best = new_metric > best_metric
                    else:
                        is_new_best = new_metric < best_metric
                    if is_new_best:
                        valid_best_metrics = deepcopy(valid_metrics)
                        save_names.append("valid_best")

                if global_step % conf.save_step == 0:
                    ckpt_dirs = [
                        key for key in os.listdir(train_dir) if key.startswith("step_")
                    ]
                    ckpt_dirs.sort(key=lambda stem: int(stem.split("_")[-1]))
                    if (
                        conf.keep_num_ckpts is not None
                        and len(ckpt_dirs) >= conf.keep_num_ckpts
                    ):
                        for ckpt_dir in ckpt_dirs[
                            : len(ckpt_dirs) - conf.keep_num_ckpts + 1
                        ]:
                            shutil.rmtree(train_dir / ckpt_dir)

                    save_names.append(f"step_{global_step}")

                for name in save_names:
                    training_stats = dict(
                        global_step=global_step,
                        epoch=epoch,
                        valid_best_metrics=valid_best_metrics,
                    )
                    _save_ckpts_to_dir(
                        train_dir / name,
                        (
                            task.module
                            if isinstance(task, _DistributedDataParallel)
                            else task
                        ),
                        optimizer,
                        scheduler,
                        build_model_all_args,
                        build_task_all_args_except_model,
                        save_model,
                        save_task,
                        training_stats,
                        global_config,
                    )

                pbar.update(1)
            epoch += 1

        pbar.close()
        if rank == 0:
            tf_logger.close()

    def evaluate(
        self,
        evaluate: dict,
        mode: str,
        task,
        dataset,
        batch_sampler,
        collate_fn,
        eval_batch: int,
        dump_dir: str,
        device: str,
        num_workers: int,
    ):
        """
        The evaluate routine used by :obj:`train` (during validation phase) and :obj:`run`
        (during testing phase).

        Args:
            evaluate (dict): same in :obj:`default_config`, no argument supported for now
            **others:
                only meaningful when you want to override this train method, which is not the
                common case. Hence we skip the documentation for now.

        """
        assert mode in ["valid", "test"]

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        task = task.to(device)
        with torch.no_grad():
            batch_results = []
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc=mode, total=len(dataloader))
            ):
                if batch_idx == eval_batch:
                    break
                batch = _to_device(batch, device)
                task.eval()
                loss, cacheable = task(mode, _dump_dir=dump_dir, **batch)
                batch_results.append(_force_cacheable(cacheable))

        logs = task.reduction(mode, batch_results, _dump_dir=dump_dir)
        return logs

    def save_model(
        self,
        save_model: dict,
        model_ckpt_dir: str,
        build_model_all_args: dict,
        model: torch.nn.Module,
    ):
        """
        Save the model state_dict and the model initialization arguments into the given directory.
        If you override this method, it is highly possible you also need to override :obj:`load_model`

        Args:
            save_model (dict): same in :obj:`default_config`, so the user can save additional settings,
                like the configuration of the dataset by duplicating the dataset hypers
                inside the :code:`save_model` field. You can rely on the :code:`omegaconf`
                package to simplify the duplication.
            model_ckpt_dir (str): save the model into the this directory.
            build_model_all_args (dict): all the arguments of :obj:`build_model`.
                By saving this dictionary, you can easily reconstruct the same model
                by calling :obj:`build_model` with the saved dictionary.
            model (torch.nn.Module): the model to be saved.

        Returns:
            None
        """
        model_ckpt_dir: Path = Path(model_ckpt_dir)
        if model_ckpt_dir.is_dir():
            shutil.rmtree(model_ckpt_dir, ignore_errors=True)
        model_ckpt_dir.mkdir(exist_ok=True, parents=True)

        with (model_ckpt_dir / "problem_name").open("w") as f:
            f.write(f"{self.__class__.__name__}")

        torch.save(model.state_dict(), model_ckpt_dir / "state_dict.pt")

        # NOTE: all arguments for building model should be in simple types (yaml serializable)
        with (model_ckpt_dir / f"arguments.yaml").open("w") as f:
            yaml.safe_dump(build_model_all_args, f)

        if len(save_model) > 0:
            with (model_ckpt_dir / "extra_conf.yaml").open("w") as f:
                yaml.safe_dump(save_model, f)

    def load_model(self, model_ckpt_dir: str):
        """
        Return the saved model.

        Args:
            model_ckpt_dir (str): Restore the model with :obj:`build_model` and the checkpoint
                saved in this directory.

        Return:
            :obj:`torch.nn.Module`
        """
        model_ckpt_dir: Path = Path(model_ckpt_dir)

        with (model_ckpt_dir / "arguments.yaml").open("r") as f:
            arguments = yaml.load(f, Loader=yaml.SafeLoader)
        model = self.build_model(**arguments)

        state_dict = torch.load(model_ckpt_dir / "state_dict.pt", map_location="cpu")
        model.load_state_dict(state_dict)

        return model

    def save_task(
        self,
        save_task: dict,
        task_ckpt_dir: str,
        build_task_all_args_except_model: dict,
        task: Task,
    ):
        """
        Save the task's state, :code:`task.get_state()`, and the initialization arguments into the given
        directory. If you override this method, it is highly possible you also need to override
        :obj:`load_task`.

        Args:
            save_task (dict): same in :obj:`default_config`, so the user can save additional settings,
                like the configuration of the dataset by duplicating the dataset hypers
                inside the :code:`save_task` field. You can rely on the :code:`omegaconf`
                package to simplify the duplication.
            task_ckpt_dir (str): save the task into this directory.
            build_task_all_args_except_model (dict): all the arguments of :obj:`build_task` except
                the :code:`model` argument since the model should be sapartely saved by
                :obj:`save_model`. By saving this dictionary, you can easily reconstruct the same task
                by calling :obj:`build_task` with the saved dictionary.
            task (Task): the task to be saved.

        Returns:
            None
        """
        task_ckpt_dir: Path = Path(task_ckpt_dir)
        if task_ckpt_dir.is_dir():
            shutil.rmtree(task_ckpt_dir, ignore_errors=True)
        task_ckpt_dir.mkdir(exist_ok=True, parents=True)

        with (task_ckpt_dir / "problem_name").open("w") as f:
            f.write(f"{self.__class__.__name__}")

        torch.save(task.get_state(), task_ckpt_dir / "state.pt")

        # NOTE: each argument is saved independently to prevent SPOF
        # i.e. a single argument which cannot be loaded will lead to missing
        # all other arguments, since the single argument file cannot be loaded
        arguments = build_task_all_args_except_model
        arguments_dir = task_ckpt_dir / "arguments"
        arguments_dir.mkdir(exist_ok=True, parents=True)
        for k, v in arguments.items():
            try:
                # If the object is yaml serializable, use yaml for readibility
                yaml.safe_dump(v)
            except:
                # If not, use pickle
                with (arguments_dir / f"{k}.pkl").open("wb") as f:
                    pickle.dump(v, f)
            else:
                with (arguments_dir / f"{k}.yaml").open("w") as f:
                    yaml.safe_dump(v, f)

        if len(save_task) > 0:
            with (task_ckpt_dir, "extra_conf.yaml").open("w") as f:
                yaml.safe_dump(save_task, f)

    def load_task(
        self, task_ckpt_dir: str, model: torch.nn.Module, task_overrides: dict = None
    ):
        """
        Return the saved task.

        Args:
            task_ckpt_dir (str): Restore the task with :obj:`build_task` and the checkpoint
                saved in this directory.
            model (torch.nn.Module): the model for the task, since the model is separately saved
                and is required for :obj:`build_task`.
            task_overrides (dict): overrides the saved initialization arguments, so can change
                the loaded task's behavior. Like, change the decoding hyperparameters.

        Returns:
            :obj:`s3prl.task.Task`
        """

        task_ckpt_dir: Path = Path(task_ckpt_dir)
        task_overrides = task_overrides or {}

        arguments = task_overrides.copy()
        arguments_dir = task_ckpt_dir / "arguments"
        for filename in os.listdir(arguments_dir):
            filepath = arguments_dir / filename
            key = filepath.stem

            if key in task_overrides:
                # do not load the file (potential crash) if the overrides already has the value
                continue

            if filepath.suffix == ".yaml":
                with filepath.open("r") as f:
                    value = yaml.load(f, Loader=yaml.SafeLoader)
            elif filepath.suffix == ".pkl":
                with filepath.open("rb") as f:
                    value = pickle.load(f)

            assert key not in arguments, (
                f"Unexpected duplicated file stem '{key}' found in {arguments_dir}. "
                "Please delete one of them."
            )
            arguments[key] = value

        task = self.build_task(model=model, **arguments)

        state = torch.load(Path(task_ckpt_dir) / "state.pt", map_location="cpu")
        task.set_state(state)

        return task

    def load_model_and_task(self, ckpts_dir: str, task_overrides: dict = None):
        """
        This is a helper method to combine :obj:`load_model` and :obj:`load_task`
        together to directly load the model and the task. This method assumes
        the model is saved under :code:`ckpts_dir / 'model'` and the task is
        saved under :code:`ckpts_dir / 'task'`

        Returns:
            tuple

            1. model (:obj:`torch.nn.Module`)
            2. task (:obj:`s3prl.task.Task`)
        """
        ckpts_dir: Path = Path(ckpts_dir)
        task_overrides = task_overrides or {}

        model = self.load_model(ckpts_dir / "model")
        task = self.load_task(ckpts_dir / "task", model, task_overrides)
        return model, task

    @staticmethod
    def _get_current_arguments(
        exclude_self_and_cls: bool = True, flatten_dict: Union[str, List[str]] = None
    ) -> dict:
        if isinstance(flatten_dict, str):
            flatten_dict = [flatten_dict]

        frame = inspect.currentframe().f_back
        args, _, _, values = inspect.getargvalues(frame)
        config = {key: values[key] for key in args}

        if exclude_self_and_cls:
            config.pop("self", None)
            config.pop("cls", None)

        if flatten_dict is not None:
            flatten_config = {}
            for k, v in config.items():
                if k in flatten_dict:
                    assert isinstance(v, dict)
                    for _k, _v in v.items():
                        flatten_config[_k] = _v
                else:
                    flatten_config[k] = v
            config = flatten_config

        def assert_no_missing(config: dict):
            omegaconf.OmegaConf.to_container(
                omegaconf.OmegaConf.create(config), throw_on_missing=True
            )

        assert_no_missing(config)
        return config

    @staticmethod
    def _get_time_tag():
        return datetime.fromtimestamp(time()).strftime("%Y_%m_%d_%H_%M_%S")

    @staticmethod
    def _stage_check(stage_id: int, stop: int, check_fn: callable):
        try:
            check_fn()
        except:
            logger.error(
                f"Stage {stage_id} was not done before or is corrupted. Please re-run from this stage."
            )
            raise
        if isinstance(stop, int) and stage_id == stop:
            exit(0)

    def main(self, args: List[str] = None):
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
            print(f"\nDefault config of {self}\n")
            print(yaml.safe_dump(self.default_config()))
            exit(0)

        if args.dump_config is not None:
            with open(args.dump_config, "w") as f:
                yaml.safe_dump(self.default_config(), f)
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

        schema = omegaconf.OmegaConf.create(self.default_config())
        config = omegaconf.OmegaConf.merge(schema, yaml_conf, override_conf)
        config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        logger.info(config)

        self.run(**config)
        return config
