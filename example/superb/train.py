import argparse
import logging
import math
import os
from copy import deepcopy
from pathlib import Path

import torch
from tqdm import tqdm

from s3prl import Container, Logs, Object, Output
from s3prl.dataset.base import AugmentedDynamicItemDataset, DataLoader
from s3prl.nn import S3PRLUpstream, UpstreamDownstreamModel
from s3prl.sampler import DistributedBatchSamplerWrapper
from s3prl.util.configuration import parse_override, qualname_to_cls
from s3prl.util.seed import fix_random_seeds

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)

DRYRUN_CONFIG = dict(
    Trainer=dict(
        total_steps=10,
        log_step=2,
        valid_step=5,
        save_step=5,
        eval_batch=5,
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", help="The upstream name. E.g. wav2vec2")
    parser.add_argument(
        "problem",
        help="The problem module. E.g. s3prl.problem.SuperbSID",
    )
    parser.add_argument(
        "dataset_root",
        help="The dataset root of your problem.",
    )
    parser.add_argument("save_to", help="The directory to save checkpoint")
    parser.add_argument("--feature_selection", default="hidden_states")
    parser.add_argument("--n_jobs", type=int, default=6)
    parser.add_argument(
        "--override",
        default=None,
        help=(
            "Override the default_config of the problem module. "
            "E.g. --override ValidSampler.batch_size=4,,TestSampler.batch_size=4"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    fix_random_seeds(args.seed)
    problem = qualname_to_cls(args.problem)
    config = Container(deepcopy(problem.default_config))

    for key, value in vars(args).items():
        if key not in ["override"]:
            config[key] = value

    if args.dryrun:
        config.override(DRYRUN_CONFIG)

    if isinstance(args.override, str) and len(args.override) > 0:
        override_dict = parse_override(args.override)
        config.override(override_dict)

    return problem, config


def main():
    logging.basicConfig(level=logging.INFO)

    problem, config = parse_args()
    save_to = Path(config.save_to)
    save_to.mkdir(exist_ok=True, parents=True)

    # configure any upstream
    upstream = S3PRLUpstream(config.upstream, config.feature_selection)
    stats = Container(upstream_rate=upstream.downsample_rate)

    logger.info("Preparing corpus")
    corpus = problem.Corpus(config.dataset_root, **config.Corpus)
    train_data, valid_data, test_data, corpus_stats = corpus().split(3)
    stats.add(corpus_stats)

    logger.info("Preparing train data")
    train_dataset = AugmentedDynamicItemDataset(train_data, tools=stats)
    train_dataset = problem.TrainData(**config.TrainData)(train_dataset)
    train_sampler = DistributedBatchSamplerWrapper(
        problem.TrainSampler(train_dataset, **config.TrainSampler),
        num_replicas=1,
        rank=0,
    )
    train_dataloader = DataLoader(
        train_dataset,
        train_sampler,
        num_workers=config.n_jobs,
    )
    stats.add(train_dataset.all_tools())

    logger.info("Preparing valid data")
    valid_dataset = AugmentedDynamicItemDataset(valid_data, tools=stats)
    valid_dataset = problem.ValidData(**config.ValidData)(valid_dataset)
    valid_sampler = DistributedBatchSamplerWrapper(
        problem.ValidSampler(valid_dataset, **config.ValidSampler),
        num_replicas=1,
        rank=0,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        valid_sampler,
        num_workers=12,
    )

    logger.info("Preparing test data")
    test_dataset = AugmentedDynamicItemDataset(test_data, tools=stats)
    test_dataset = problem.TestData(**config.TestData)(test_dataset)
    test_sampler = DistributedBatchSamplerWrapper(
        problem.ValidSampler(test_dataset, **config.TestSampler),
        num_replicas=1,
        rank=0,
    )
    test_dataloader = DataLoader(
        test_dataset,
        test_sampler,
        num_workers=12,
    )

    sorted_ckpt_dirs = sorted(
        [
            file
            for file in save_to.iterdir()
            if file.is_dir() and str(file).endswith(".ckpts")
        ],
        key=os.path.getmtime,
    )

    if config.resume and len(sorted_ckpt_dirs) > 0:
        logger.info("Last checkpoint found. Load model and optimizer from checkpoint")
        task = Object.load_checkpoint(sorted_ckpt_dirs[1] / "task.ckpt").to(device)
    else:
        logger.info("Create a new model")
        downstream = problem.Downstream(
            upstream.output_size,
            **stats,
        )
        model = UpstreamDownstreamModel(upstream, downstream)
        # task = problem.Task(model, **{**stats, **config.Task})
        task = problem.Task(model, **stats, **config.Task)
        task = task.to(device)

    # ALL THE FOLLOWING CODES ARE FOR TRAINER
    # WHICH CAN BE LARGELY SIMPLIFIED WHEN USING OTHER TRAINER PACKAGES

    opt_cls_qualname, opt_cfgs = config.Optimizer.split(1)
    optimizer = qualname_to_cls(opt_cls_qualname)(task.parameters(), **opt_cfgs)
    if config.resume and len(sorted_ckpt_dirs) > 0:
        optimizer.load_state_dict(torch.load(sorted_ckpt_dirs[-1] / "optimizer.ckpt"))

    if config.Trainer.use_valid:
        if config.resume and len(sorted_ckpt_dirs) > 0:
            valid_best_score = torch.load(
                sorted_ckpt_dirs[-1] / "valid_best_score.ckpt"
            )[config.Trainer.valid_metric]
        else:
            valid_best_score = -100000 if config.Trainer.valid_higher_better else 100000

    def save_checkpoint(name):
        ckpt_dir: Path = save_to / f"{name}.ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save checkpoint to: {ckpt_dir}")

        task.save_checkpoint(ckpt_dir / "task.ckpt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.ckpt")
        torch.save(
            {config.Trainer.valid_metric: valid_best_score},
            ckpt_dir / "valid_best_score.ckpt",
        )

    pbar = tqdm(total=config.Trainer.total_steps, desc="Total")
    train_completed = False
    accum_grad_steps = 0
    while not train_completed:
        batch_results = []
        for batch in tqdm(train_dataloader, desc="Train", total=len(train_dataloader)):
            pbar.update(1)
            global_step = pbar.n

            assert isinstance(batch, Output)
            batch = batch.to(device)

            task.train()
            result = task.train_step(**batch)
            assert isinstance(result, Output)

            result.loss /= config.Trainer.gradient_accumulate_steps
            result.loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                task.parameters(), max_norm=config.Trainer.gradient_clipping
            )

            if math.isnan(grad_norm):
                logger.warning(f"Grad norm is NaN at step {global_step}")
                optimizer.zero_grad()
                accum_grad_steps = 0
            else:
                accum_grad_steps += 1
                if accum_grad_steps == config.Trainer.gradient_accumulate_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_grad_steps = 0
                batch_results.append(result.cacheable())

            if global_step % config.Trainer.log_step == 0:
                logs: Logs = task.train_reduction(batch_results).logs
                logger.info(f"[Train] step {global_step}")
                for name, value in logs.Scalar.items():
                    if name == "loss":
                        value *= config.Trainer.gradient_accumulate_steps
                    logger.info(f"{name}: {value}")
                batch_results = []

            if global_step % config.Trainer.valid_step == 0:
                with torch.no_grad():
                    if config.Trainer.use_valid:
                        valid_results = []
                        for batch_idx, batch in enumerate(
                            tqdm(
                                valid_dataloader,
                                desc="Valid",
                                total=len(valid_dataloader),
                            )
                        ):
                            if batch_idx == config.Trainer.get("eval_batch", -1):
                                break
                            batch = batch.to(device)
                            task.eval()
                            result = task.valid_step(**batch)
                            valid_results.append(result.cacheable())

                        logs: Logs = task.valid_reduction(valid_results).slice(1)
                        logger.info(f"[Valid] step {global_step}")
                        for name, value in logs.Scalar.items():
                            logger.info(f"{name}: {value}")
                            if name == config.Trainer.valid_metric:
                                cond1 = config.Trainer.valid_higher_better and (
                                    value > valid_best_score
                                )
                                cond2 = (not config.Trainer.valid_higher_better) and (
                                    value < valid_best_score
                                )
                                if cond1 or cond2:
                                    valid_best_score = value
                                    save_checkpoint("valid_best")

            if (
                global_step % config.Trainer.save_step == 0
                or global_step == config.Trainer.total_steps
            ):
                save_checkpoint(f"global_step_{global_step}")

                if global_step == config.Trainer.total_steps:
                    train_completed = True
                    break

    test_results = []
    for batch_idx, batch in enumerate(
        tqdm(test_dataloader, desc="Test", total=len(test_dataloader))
    ):
        if batch_idx == config.Trainer.get("eval_batch", -1):
            break
        batch = batch.to(device)
        result = task.test_step(**batch)
        test_results.append(result.cacheable())

    logs: Logs = task.test_reduction(test_results).slice(1)
    logger.info(f"[Test] step {global_step}")
    for name, value in logs.Scalar.items():
        logger.info(f"{name}: {value}")


if __name__ == "__main__":
    main()
