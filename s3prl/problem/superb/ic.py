from __future__ import annotations

import torch.nn as nn

from s3prl import Container, field
from s3prl.corpus.fluent_speech_commands import (
    FluentSpeechCommandsForUtteranceMultiClassClassificataion,
)
from s3prl.dataset.utterance_classification_pipe import (
    UtteranceMultipleCategoryClassificationPipe,
)
from s3prl.nn import MeanPoolingLinear
from s3prl.problem.trainer import Trainer
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.utterance_classification_task import (
    UtteranceMultiClassClassificationTask,
)
from s3prl.util.configuration import override_parent_cfg

from .base import SuperbProblem


class SuperbIC(SuperbProblem):
    """
    Superb Intent Classification problem
    """

    @override_parent_cfg(
        corpus=dict(
            _cls=FluentSpeechCommandsForUtteranceMultiClassClassificataion,
            dataset_root="???",
        ),
        train_datapipe=dict(
            _cls=UtteranceMultipleCategoryClassificationPipe,
            train_category_encoder=True,
        ),
        train_sampler=dict(
            _cls=MaxTimestampBatchSampler,
            max_timestamp=16000 * 200,
            shuffle=True,
        ),
        valid_datapipe=dict(
            _cls=UtteranceMultipleCategoryClassificationPipe,
        ),
        valid_sampler=dict(
            _cls=FixedBatchSizeBatchSampler,
            batch_size=2,
        ),
        test_datapipe=dict(
            _cls=UtteranceMultipleCategoryClassificationPipe,
        ),
        test_sampler=dict(
            _cls=FixedBatchSizeBatchSampler,
            batch_size=2,
        ),
        downstream=dict(
            _cls=MeanPoolingLinear,
            hidden_size=256,
        ),
        task=dict(
            _cls=UtteranceMultiClassClassificationTask,
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
        ),
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

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
        )
    )
    @classmethod
    def run_stages(cls, **cfg):
        super().run_stages(**cfg)
