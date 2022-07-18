from __future__ import annotations

import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from s3prl import Container, field
from s3prl.base.logdata import Logs
from s3prl.corpus.voxceleb1sv import voxceleb1_for_sv
from s3prl.dataset.base import DataLoader
from s3prl.dataset.speaker_verification_pipe import SpeakerVerificationPipe
from s3prl.nn.speaker_model import SuperbXvector
from s3prl.sampler import FixedBatchSizeBatchSampler
from s3prl.task.speaker_verification_task import SpeakerVerification
from s3prl.util.configuration import default_cfg
from s3prl.util.workspace import Workspace

from .base import SuperbProblem

logger = logging.getLogger(__name__)


EFFECTS = [
    ["channels", "1"],
    ["rate", "16000"],
    ["gain", "-3.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]


class SuperbSV(SuperbProblem):
    """
    Superb Speaker Verification problem
    """

    @default_cfg(
        **SuperbProblem.setup.default_except(
            corpus=dict(
                CLS=voxceleb1_for_sv,
                dataset_root="???",
            ),
            train_datapipe=dict(
                CLS=SpeakerVerificationPipe,
                random_crop_secs=8.0,
                sox_effects=EFFECTS,
            ),
            train_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=10,
                shuffle=True,
            ),
            valid_datapipe=dict(
                CLS=SpeakerVerificationPipe,
                sox_effects=EFFECTS,
            ),
            valid_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=1,
            ),
            test_datapipe=dict(
                CLS=SpeakerVerificationPipe,
                sox_effects=EFFECTS,
            ),
            test_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=1,
            ),
            downstream=dict(
                CLS=SuperbXvector,
            ),
            task=dict(
                CLS=SpeakerVerification,
                loss_type="amsoftmax",
                loss_cfg=dict(
                    margin=0.4,
                    scale=30,
                ),
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        """
        This setups the ASV problem, containing train/valid/test datasets & samplers and a task object
        """
        super().setup(**cfg)

    @default_cfg(
        **SuperbProblem.train.default_except(
            optimizer=dict(
                CLS="torch.optim.AdamW",
                lr=1.0e-4,
            ),
            trainer=dict(
                total_steps=200000,
                log_step=500,
                eval_step=field(1e10, "ASV do not use validation set"),
                save_step=20000,
                gradient_clipping=1.0e3,
                gradient_accumulate_steps=5,
                valid_metric="eer",
                valid_higher_better=False,
                max_keep=10,
            ),
        )
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

    @default_cfg(
        **SuperbProblem.inference.default_except(
            inference_steps=field(
                [
                    20000,
                    40000,
                    60000,
                    80000,
                    100000,
                    120000,
                    140000,
                    160000,
                    180000,
                    200000,
                ],
                "The steps used for inference\n",
                "egs: [900, 1000] - use the checkpoint of 90 and 100 steps for inference",
            )
        )
    )
    @classmethod
    def inference(cls, **cfg):
        cfg = Container(cfg)

        workspace = Workspace(cfg.workspace)
        dataset = workspace[f"{cfg.split_name}_dataset"]
        sampler = workspace[f"{cfg.split_name}_sampler"]
        dataloader = DataLoader(dataset, sampler, num_workers=cfg.n_jobs)

        with torch.no_grad():
            all_eers = []
            for step in cfg.inference_steps:
                step_dir = workspace / f"step-{step}"
                task = step_dir["task"]
                task = task.to(cfg.device)
                task.eval()

                test_results = []
                for batch_idx, batch in enumerate(
                    tqdm(dataloader, desc="Test", total=len(dataloader))
                ):
                    batch = batch.to(cfg.device)
                    result = task.test_step(**batch)
                    test_results.append(result.cacheable())

                logs: Logs = task.test_reduction(test_results).logs
                logger.info(f"Step {step}")

                metrics = {key: value for key, value in logs.scalars()}
                step_dir.put(metrics, "test_metrics", "yaml")
                for key, value in metrics.items():
                    logger.info(f"{key}: {value}")
                all_eers.append(metrics["EER"])

            workspace.put({"minEER": min(all_eers)}, "test_metrics", "yaml")

    @default_cfg(
        **SuperbProblem.run.default_except(
            stages=["setup", "train", "inference"],
            start_stage="setup",
            final_stage="inference",
            setup=setup.default_cfg.deselect("workspace", "resume", "dryrun"),
            train=train.default_cfg.deselect("workspace", "resume", "dryrun"),
            inference=inference.default_cfg.deselect("workspace", "resume", "dryrun"),
        )
    )
    @classmethod
    def run(cls, **cfg):
        super().run(**cfg)
