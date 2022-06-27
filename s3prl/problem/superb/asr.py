from s3prl.corpus.librispeech import librispeech_for_speech2text
from s3prl.dataset.speech2text_pipe import Speech2TextPipe
from s3prl.nn import RNNEncoder
from s3prl.nn.specaug import ModelWithSpecaug
from s3prl.sampler import FixedBatchSizeBatchSampler
from s3prl.task.speech2text_ctc_task import Speech2TextCTCTask
from s3prl.util.configuration import default_cfg

from .base import SuperbProblem


class SuperbASR(SuperbProblem):
    @default_cfg(
        **SuperbProblem.setup.default_except(
            corpus=dict(
                _cls=librispeech_for_speech2text,
                dataset_root="???",
            ),
            train_datapipe={
                "0": dict(
                    _cls=Speech2TextPipe,
                ),
            },
            train_sampler=dict(
                _cls=FixedBatchSizeBatchSampler,
                batch_size=32,
                shuffle=True,
            ),
            valid_datapipe={
                "0": dict(
                    _cls=Speech2TextPipe,
                )
            },
            valid_sampler=dict(
                _cls=FixedBatchSizeBatchSampler,
                batch_size=1,
            ),
            test_datapipe={
                "0": dict(
                    _cls=Speech2TextPipe,
                )
            },
            test_sampler=dict(
                _cls=FixedBatchSizeBatchSampler,
                batch_size=1,
            ),
            downstream=dict(
                _cls=ModelWithSpecaug,
                model_cfg=dict(
                    _cls=RNNEncoder,
                    module="LSTM",
                    proj_size=1024,
                    hidden_size=[1024, 1024],
                    dropout=[0.2, 0.2],
                    layer_norm=[False, False],
                    proj=[False, False],
                    sample_rate=[1, 1],
                    sample_style="concat",
                    bidirectional=True,
                ),
                specaug_cfg=dict(
                    freq_mask_width_range=(0, 50),
                    num_freq_mask=4,
                    time_mask_width_range=(0, 40),
                    num_time_mask=2,
                ),
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
                lr=1.0e-4,
            ),
            trainer=dict(
                total_steps=200000,
                log_step=100,
                eval_step=2000,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate_steps=1,
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
            setup=setup.default_cfg.deselect("workspace", "resume"),
            train=train.default_cfg.deselect("workspace", "resume"),
            inference=inference.default_cfg.deselect("workspace", "resume"),
        )
    )
    @classmethod
    def run_stages(cls, **cfg):
        super().run_stages(**cfg)
