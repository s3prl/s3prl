from s3prl import Container
from s3prl.corpus.speech_commands import gsc_v1_for_superb
from s3prl.dataset.utterance_classification_pipe import UtteranceClassificationPipe
from s3prl.nn import MeanPoolingLinear
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.utterance_classification_task import UtteranceClassificationTask
from s3prl.util.configuration import default_cfg

from .base import SuperbProblem


class SuperbKS(SuperbProblem):
    @default_cfg(
        **SuperbProblem.setup.default_except(
            corpus=dict(
                _cls=gsc_v1_for_superb,
                dataset_root="???",
            ),
            train_datapipe={
                "0": dict(
                    _cls=UtteranceClassificationPipe,
                    train_category_encoder=True,
                ),
            },
            train_sampler=dict(
                _cls=FixedBatchSizeBatchSampler,
                batch_size=32,
                shuffle=True,
            ),
            valid_datapipe={
                "0": dict(
                    _cls=UtteranceClassificationPipe,
                ),
            },
            valid_sampler=dict(
                _cls=FixedBatchSizeBatchSampler,
                batch_size=32,
            ),
            test_datapipe={
                "0": dict(
                    _cls=UtteranceClassificationPipe,
                ),
            },
            test_sampler=dict(
                _cls=FixedBatchSizeBatchSampler,
                batch_size=32,
            ),
            downstream=dict(
                _cls=MeanPoolingLinear,
                hidden_size=256,
            ),
            task=dict(
                _cls=UtteranceClassificationTask,
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        super().setup(**cfg)

    @default_cfg(
        **SuperbProblem.train.default_except(
            optimizer=dict(
                _cls="torch.optim.Adam",
                lr=1.0e-4,
            ),
            trainer=dict(
                total_steps=200000,
                log_step=100,
                eval_step=5000,
                save_step=1000,
                gradient_clipping=1.0,
                gradient_accumulate_steps=1,
                valid_metric="accuracy",
                valid_higher_better=True,
            ),
        )
    )
    @classmethod
    def train(cls, **cfg):
        super().train(**cfg)

    @default_cfg(**SuperbProblem.inference.default_cfg)
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @default_cfg(
        stages=["setup", "train", "inference"],
        start_stage="setup",
        final_stage="inference",
        setup=setup.default_cfg.deselect("workspace", "resume"),
        train=train.default_cfg.deselect("workspace", "resume"),
        inference=inference.default_cfg.deselect("workspace", "resume"),
    )
    @classmethod
    def run_stages(cls, **cfg):
        super().run_stages(**cfg)
