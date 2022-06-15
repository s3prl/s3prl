from s3prl.corpus.librispeech import librispeech_for_speech2text
from s3prl.dataset.speech2phoneme_pipe import Speech2PhonemePipe
from s3prl.nn.linear import FrameLevelLinear
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.speech2text_ctc_task import Speech2TextCTCTask
from s3prl.util.configuration import default_cfg

from .base import SuperbProblem


class SuperbPR(SuperbProblem):
    @default_cfg(
        **SuperbProblem.setup.default_except(
            corpus=dict(
                _cls=librispeech_for_speech2text,
                dataset_root="???",
            ),
            train_datapipe=dict(
                _cls=Speech2PhonemePipe,
            ),
            train_sampler=dict(
                _cls=FixedBatchSizeBatchSampler,
                batch_size=8,
                shuffle=True,
            ),
            valid_datapipe=dict(
                _cls=Speech2PhonemePipe,
            ),
            valid_sampler=dict(
                _cls=FixedBatchSizeBatchSampler,
                batch_size=8,
            ),
            test_datapipe=dict(
                _cls=Speech2PhonemePipe,
            ),
            test_sampler=dict(
                _cls=FixedBatchSizeBatchSampler,
                batch_size=8,
            ),
            downstream=dict(
                _cls=FrameLevelLinear,
            ),
            task=dict(
                _cls=Speech2TextCTCTask,
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
                lr=1.0e-2,
            ),
            trainer=dict(
                total_steps=100000,
                log_step=100,
                eval_step=1000,
                save_step=100,
                gradient_clipping=1.0,
                gradient_accumulate_steps=2,
                valid_metric="wer",
                valid_higher_better=False,
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
        **SuperbProblem.run_stages.default_except(
            stages=["setup", "train", "inference"],
            start_stage="setup",
            final_stage="inference",
            setup=setup.default_cfg.deselect("workspace", "resume", "dryrun"),
            train=train.default_cfg.deselect("workspace", "resume", "dryrun"),
            inference=inference.default_cfg.deselect("workspace", "resume", "dryrun"),
        )
    )
    @classmethod
    def run_stages(cls, **cfg):
        super().run_stages(**cfg)
