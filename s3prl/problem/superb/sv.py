from __future__ import annotations

import logging

import torch.nn as nn
from tqdm import tqdm

from s3prl import Container, field
from s3prl.base.logdata import Logs
from s3prl.corpus.voxceleb1sv import VoxCeleb1SV
from s3prl.dataset.base import DataLoader
from s3prl.dataset.speaker_verification_pipe import SpeakerClassificationPipe
from s3prl.nn import speaker_embedding_extractor
from s3prl.problem.trainer import Trainer
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.speaker_verification_task import SpeakerVerification
from s3prl.util.configuration import override_parent_cfg
from s3prl.util.workspace import Workspace

from .base import SuperbProblem

logger = logging.getLogger(__name__)


class SuperbSV(SuperbProblem):
    """
    Superb Speaker Verification problem
    """

    @override_parent_cfg(
        corpus=dict(
            _cls=VoxCeleb1SV,
            dataset_root="???",
        ),
        train_datapipe=dict(
            _cls=SpeakerClassificationPipe,
            train_category_encoder=True,
        ),
        train_sampler=dict(
            _cls=MaxTimestampBatchSampler,
            max_timestamp=16000 * 200,
            shuffle=True,
        ),
        valid_datapipe=dict(
            _cls=SpeakerClassificationPipe,
        ),
        valid_sampler=dict(
            _cls=FixedBatchSizeBatchSampler,
            batch_size=2,
        ),
        test_datapipe=dict(
            _cls=SpeakerClassificationPipe,
        ),
        test_sampler=dict(
            _cls=FixedBatchSizeBatchSampler,
            batch_size=2,
        ),
        downstream=dict(
            _cls=speaker_embedding_extractor,
            hidden_size=256,
        ),
        task=dict(
            _cls=SpeakerVerification,
        ),
    )
    @classmethod
    def setup_problem(cls, **cfg):
        """
        This setups the IC problem, containing train/valid/test datasets & samplers and a task object
        """
        super().setup_problem(**cfg)

    @override_parent_cfg(
        optimizer=dict(
            _cls="torch.optim.Adam",
            lr=1.0e-4,
        ),
        trainer=dict(
            total_steps=1000,
            log_step=100,
            eval_step=500,
            save_step=100,
            gradient_clipping=1.0,
            gradient_accumulate_steps=4,
            valid_metric="accuracy",
            valid_higher_better=True,
            max_keep=2,
        ),
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

    @override_parent_cfg(
        inference_steps=field(
            "???",
            "The steps used for inference\n",
            "egs: 900,,1000 - use the checkpoint of 90 and 100 steps for inference",
        )
    )
    @classmethod
    def inference(cls, **cfg):
        cfg = Container(cfg)
        if cfg.dryrun:
            cfg.override(cls.INFERENCE_DRYRUN_CONFIG)

        workspace = Workspace(cfg.workspace)
        dataset = workspace[f"{cfg.split_name}_dataset"]
        sampler = workspace[f"{cfg.split_name}_sampler"]
        dataloader = DataLoader(dataset, sampler, num_workers=cfg.n_jobs)

        inference_steps = cfg.inference_steps.split(",,")
        for step in inference_steps:

            step_dir = workspace / f"step-{step}"
            task = step_dir["task"]
            task = task.to(cfg.device)

            test_results = []
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Test", total=len(dataloader))
            ):
                batch = batch.to(cfg.device)
                result = task.test_step(**batch)
                test_results.append(result.cacheable())

            logs: Logs = task.test_reduction(test_results).logs
            logger.info(f"[Test] - Step {step}")

            for key in logs.keys():
                logger.info(f"{key}: {logs[key].data}")

    @override_parent_cfg(
        start_stage=0,
        final_stage=2,
        stage_0=dict(
            _method="setup_problem",
        ),
        stage_1=dict(
            _method="train",
        ),
        stage_2=dict(
            _method="inference",
        ),
    )
    @classmethod
    def run_stages(cls, **cfg):
        super().run_stages(**cfg)
