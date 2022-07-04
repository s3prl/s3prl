import logging
import math
import os
import sys

import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

from s3prl import Output
from s3prl.base.container import Container, field
from s3prl.base.logdata import Logs
from s3prl.dataset.base import DataLoader
from s3prl.problem.base import Problem
from s3prl.sampler.distributed_sampler import DistributedBatchSamplerWrapper
from s3prl.task import Task
from s3prl.util.configuration import default_cfg
from s3prl.util.seed import fix_random_seeds
from s3prl.util.workspace import Workspace

logger = logging.getLogger(__name__)

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


class Trainer:
    @default_cfg(
        workspace=field(
            "???",
            "\nThe workspace should have the following keys:\n"
            "  'train_dataset', 'train_sampler', 'valid_dataset', 'valid_sampler', and 'task'\n"
            "\nWill put the following keys into this workspace:\n"
            "  'valid_best_task': the trained task\n",
            "str or Path or Workspace",
        ),
        resume=field(
            False,
            "If true, load the config of the previous run (if exists) as the default config, and the saved checkpoints if exists",
            bool,
        ),
        n_jobs=field(4, "The number of jobs when multiprocessing on CPU", int),
        seed=field(1337, "The seed", int),
        device=field("cuda:0", "The device used for training", str),
        rank=field(0, "The global rank when distributed training", int),
        world_size=field(
            1, "The total number of processes when distributed training", int
        ),
        optimizer=dict(
            _cls=field(
                torch.optim.Adam,
                "The class used to create the optimizer. The below are **kwargs for the __init__",
                str,
            ),
            lr=1.0e-4,
            betas=(0.9, 0.999),
        ),
        use_scheduler=field(False, "Whether to enable the scheduler", bool),
        scheduler=dict(
            _cls=field(
                CyclicLR,
                "The class used to create the scheduler. The below are **kwargs for the __init__",
                str,
            ),
            start_factor=0.34,
        ),
        trainer=dict(
            total_steps=field(10, "The total training steps", int),
            log_step=field(
                2,
                "How many training steps for a single logging step to log the averaged training metrics",
                int,
            ),
            save_step=field(5, "How many training steps to save a checkpoint", int),
            max_keep=field(2, "Save the last 'max_keep' number of checkpoints", int),
            eval_step=field(
                5,
                "How many training steps for an evaluation epoch on the validation set",
                int,
            ),
            eval_batch=field(
                -1,
                "Only go through 'eval_batch' steps when doing evaluation. Use -1 to disable",
                int,
            ),
            gradient_clipping=field(
                1.0, "Clipping the gradient is essential especially for RNNs", float
            ),
            gradient_accumulate_steps=field(
                1,
                "Accumulate n steps of training gradient and backward once. Useful for simulating large batch size training",
                int,
            ),
            valid_metric=field(
                "loss",
                "The metric to compare for saving the valid_best checkpoint",
                str,
            ),
            valid_higher_better=field(
                False, "Decide how to save the valid_best checkpoint", bool
            ),
        ),
    )
    @classmethod
    def train(cls, **cfg):
        cfg = Container(cfg)
        fix_random_seeds(cfg.seed)

        workspace = Workspace(cfg.workspace)
        workspace.set_rank(cfg.rank)

        train_dataset, train_sampler = workspace.gets("train_dataset", "train_sampler")
        valid_dataset, valid_sampler = workspace.gets("valid_dataset", "valid_sampler")

        ckpts = [key for key in workspace.dirs() if "step-" in key]
        if cfg.resume and len(ckpts) > 0:
            ckpts.sort(key=lambda name: int(name.split("-")[-1]), reverse=True)
            latest_ckpt = workspace / ckpts[-1]
            logger.info(f"Load checkpoint from {latest_ckpt}")

            task = latest_ckpt["task"]
            optimizer_state = latest_ckpt["optimizer_state"]
            scheduler_state = latest_ckpt.get("scheduler_state", None)
            global_step = latest_ckpt["global_step"]
            epoch = latest_ckpt["epoch"]
            valid_best_metric = latest_ckpt["valid_best_metric"]
        else:
            task = workspace["task"]
            optimizer_state = None
            scheduler_state = None
            global_step = 0
            epoch = 0
            valid_best_metric = dict()

        device = torch.device(cfg.device)
        wrapped_task = task.to(device)

        if cfg.world_size > 1:
            torch.cuda.set_device(device.index)
            wrapped_task = DistributedDataParallel(
                task,
                device_ids=[device.index],
                find_unused_parameters=True,
                output_device=device.index,
            )

        train_sampler = DistributedBatchSamplerWrapper(
            train_sampler,
            num_replicas=cfg.world_size,
            rank=cfg.rank,
        )
        train_dataloader = DataLoader(
            train_dataset,
            train_sampler,
            num_workers=cfg.n_jobs,
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            DistributedBatchSamplerWrapper(
                valid_sampler,
                num_replicas=cfg.world_size,
                rank=cfg.rank,
            ),
            num_workers=cfg.n_jobs,
        )

        optimizer = cfg.optimizer._cls(
            wrapped_task.parameters(),
            **cfg.optimizer.kwds(),
        )
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        scheduler = None
        if cfg.use_scheduler:
            scheduler = cfg.scheduler._cls(optimizer, **cfg.scheduler.kwds())
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)

        tqdm_file = sys.stderr if cfg.rank == 0 else open(os.devnull, "w")
        pbar = tqdm(
            total=cfg.trainer.total_steps,
            dynamic_ncols=True,
            desc="train",
            file=tqdm_file,
        )
        pbar.n = global_step

        if cfg.rank == 0:
            tf_dir = workspace / "tf"
            tf_logger = SummaryWriter(str(tf_dir))

        backward_steps = 0
        while pbar.n < pbar.total:
            train_sampler.set_epoch(epoch),
            batch_results = []
            logger.info(f"Start epoch {epoch}")
            for batch in train_dataloader:
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    wrapped_task.train()
                    batch = batch.to(device)
                    result: Output = wrapped_task("train", **batch)
                    # result: Output = wrapped_task.train_step(**batch)
                    (result.loss / cfg.trainer.gradient_accumulate_steps).backward()
                    batch_results.append(result.cacheable())

                except RuntimeError as e:
                    if cfg.world_size > 1:
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
                if backward_steps % cfg.trainer.gradient_accumulate_steps > 0:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    wrapped_task.parameters(), cfg.trainer.gradient_clipping
                )

                if math.isnan(grad_norm):
                    logger.warning(f"[Runner] - grad norm is NaN at step {global_step}")
                else:
                    optimizer.step()
                optimizer.zero_grad()

                if scheduler:
                    scheduler.step()

                if cfg.rank > 0:
                    batch_results = []
                    pbar.update(1)
                    continue

                if global_step % cfg.trainer.log_step == 0:
                    logs = wrapped_task.reduction("train", batch_results).logs
                    cls.log_results("train", logs, tf_logger, global_step)
                    batch_results = []

                save_names = []

                if global_step % cfg.trainer.eval_step == 0:
                    logs: Logs = cls.evaluate(
                        "valid",
                        task,
                        valid_dataloader,
                        device,
                        cfg.trainer.eval_batch,
                    )
                    cls.log_results("valid", logs, tf_logger, global_step)
                    valid_metric_key = cfg.trainer.valid_metric
                    valid_metric = Container({k: v for k, v in logs.scalars()})
                    new_metric = valid_metric[valid_metric_key]
                    best_metric = valid_best_metric.get(valid_metric_key, None)
                    if best_metric is None:
                        is_new_best = True
                    elif cfg.trainer.valid_higher_better:
                        is_new_best = new_metric > best_metric
                    else:
                        is_new_best = new_metric < best_metric
                    if is_new_best:
                        valid_best_metric = valid_metric.clone()
                        save_names.append("valid_best")

                if global_step % cfg.trainer.save_step == 0:
                    ckpt_dirs = [key for key in workspace.dirs() if "step-" in key]
                    ckpt_dirs.sort(key=lambda stem: int(stem.split("-")[-1]))
                    max_keep = cfg.trainer.max_keep
                    if len(ckpt_dirs) >= max_keep:
                        for ckpt_dir in ckpt_dirs[: len(ckpt_dirs) - max_keep + 1]:
                            workspace.remove(ckpt_dir)

                    save_names.append(f"step-{global_step}")

                for name in save_names:
                    cls.save_checkpoint(
                        workspace / name,
                        task,
                        optimizer_state,
                        scheduler_state,
                        global_step,
                        epoch,
                        valid_best_metric,
                    )
                    cls.save_additional(
                        workspace / name / "additional",
                        workspace,
                        task,
                    )
                if "valid_best" in save_names:
                    workspace.link_from(
                        "valid_best_task", workspace / "valid_best", "task"
                    )

                pbar.update(1)
            epoch += 1

        pbar.close()
        if cfg.rank == 0:
            tf_logger.close()

    @classmethod
    def log_results(
        cls,
        split_name: str,
        logs: Logs,
        tensorboard: SummaryWriter,
        global_step: int,
    ):
        logger.info(f"{split_name} at step {global_step}")
        for name, value in logs.scalars():
            logger.info(f"{name}: {value}")
            tensorboard.add_scalar(
                f"{split_name}-{name}", value, global_step=global_step
            )

    @classmethod
    def save_checkpoint(
        cls,
        checkpoint_dir: Workspace,
        task: Task,
        optimizer_state,
        scheduler_state,
        global_step,
        epoch,
        valid_best_metric,
    ):
        logger.info(f"Save checkpoint to {str(checkpoint_dir)}")
        checkpoint_dir.put(task, "task")
        checkpoint_dir.put(optimizer_state, "optimizer_state", "pt")
        checkpoint_dir.put(scheduler_state, "scheduler_state", "pt")
        checkpoint_dir.put(global_step, "global_step", "yaml")
        checkpoint_dir.put(epoch, "epoch", "yaml")
        checkpoint_dir.put(valid_best_metric, "valid_best_metric", "yaml")

    @classmethod
    def save_additional(
        cls,
        additional_dir: Workspace,
        workspace: Workspace,
        task: Task,
    ):
        # You can get the config from all previous stages and the current stage
        train_cfg = workspace.get_cfg(cls.train)
        assert isinstance(train_cfg, dict)
        assert "workspace" in train_cfg

    @classmethod
    def evaluate(
        cls,
        split_name: str,
        task: Task,
        dataloader,
        device: str = "cuda:0",
        eval_batch: int = -1,
        eval_workspace: Workspace = None,
    ):
        task = task.to(device)
        with torch.no_grad():
            batch_results = []
            for batch_idx, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=split_name,
                    total=len(dataloader),
                )
            ):
                if batch_idx == eval_batch:
                    break
                batch = batch.to(device)
                task.eval()
                result = task(split_name, **batch, workspace=eval_workspace)
                # result = task.valid_step(**batch)
                batch_results.append(result.cacheable())

        logs = task.reduction(split_name, batch_results, workspace=eval_workspace).logs
        # logs = task.valid_reduction(batch_results).logs
        return logs

    @default_cfg(
        task_name=field("valid_best_task", "The task to be inferenced", str),
        split_name=field("test", "The split of a dataset and sampler pair", str),
        workspace=field(
            "???",
            "The workspace containing the following keys:\n"
            "  - {split}_dataset: Any pytorch Dataset\n"
            "  - {split}_sampler: Any pytorch batch smapler\n"
            "  - valid_best_task: A trained Task to be inference\n"
            "\n"
            "And the function will provide:\n"
            "  - {split}_metrics: a dictionary with {metric_name: metric_value}",
            str,
        ),
        eval_batch=field(
            -1,
            "How many batches to evaluate. Use -1 to run all batches (entire epoch)",
            int,
        ),
        n_jobs=field(4, "The number of workers for the dataloader", int),
        device=field("cuda:0", "The device for inference", str),
    )
    @classmethod
    def inference(cls, **cfg):
        cfg = Container(cfg)
        workspace = Workspace(cfg.workspace)
        fix_random_seeds()

        dataset = workspace[f"{cfg.split_name}_dataset"]
        sampler = workspace[f"{cfg.split_name}_sampler"]
        dataloader = DataLoader(dataset, sampler, num_workers=cfg.n_jobs)
        task = workspace[cfg.task_name]
        logs: Logs = cls.evaluate(
            cfg.split_name,
            task,
            dataloader,
            cfg.device,
            cfg.eval_batch,
            workspace,
        )
        workspace.put(
            {k: v for k, v in logs.scalars()}, f"{cfg.split_name}_metrics", "yaml"
        )
        for key, value in logs.scalars():
            logger.info(f"{key}: {value}")
