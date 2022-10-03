from __future__ import annotations

from s3prl.corpus.fluent_speech_commands import fsc_for_multiple_classfication
from s3prl.dataset.utterance_classification_pipe import (
    UtteranceMultipleCategoryClassificationPipe,
)
from s3prl.nn import MeanPoolingLinear
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.utterance_classification_task import (
    UtteranceMultiClassClassificationTask,
)
from s3prl.util.configuration import default_cfg

from .base import SuperbProblem


class SuperbIC(SuperbProblem):
    """
    Superb Intent Classification problem
    """

    @default_cfg(
        **SuperbProblem.setup.default_except(
            corpus=dict(
                CLS=fsc_for_multiple_classfication,
                dataset_root="???",
            ),
            train_datapipe=dict(
                CLS=UtteranceMultipleCategoryClassificationPipe,
                train_category_encoder=True,
            ),
            train_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=32,
                shuffle=True,
            ),
            valid_datapipe=dict(
                CLS=UtteranceMultipleCategoryClassificationPipe,
            ),
            valid_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=32,
            ),
            test_datapipe=dict(
                CLS=UtteranceMultipleCategoryClassificationPipe,
            ),
            test_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=32,
            ),
            downstream=dict(
                CLS=MeanPoolingLinear,
                hidden_size=256,
            ),
            task=dict(
                CLS=UtteranceMultiClassClassificationTask,
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        """
        This setups the IC problem, containing train/valid/test datasets & samplers and a task object
        """
        super().setup(**cfg)

    @default_cfg(
        **SuperbProblem.train.default_except(
            optimizer=dict(
                CLS="torch.optim.Adam",
                lr=1.0e-4,
            ),
            trainer=dict(
                total_steps=200000,
                log_step=100,
                eval_step=5000,
                save_step=250,
                gradient_clipping=1.0,
                gradient_accumulate_steps=1,
                valid_metric="accuracy",
                valid_higher_better=True,
            ),
        )
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

    @default_cfg(**SuperbProblem.inference.default_cfg)
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @default_cfg(
        **SuperbProblem.run.default_except(
            stages=["setup", "train", "inference"],
            start_stage="setup",
            final_stage="inference",
            setup=setup.default_cfg.deselect("workspace", "resume"),
            train=train.default_cfg.deselect("workspace", "resume"),
            inference=inference.default_cfg.deselect("workspace", "resume"),
        )
    )
    @classmethod
    def run(cls, **cfg):
        super().run(**cfg)
