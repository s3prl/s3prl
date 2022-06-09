from s3prl.corpus.librispeech import librispeech_for_speech2text
from s3prl.dataset.speech2phoneme_pipe import Speech2PhonemePipe
from s3prl.nn import RNNEncoder
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.speech2text_ctc_task import Speech2TextCTCTask
from s3prl.util.configuration import override_parent_cfg

from .base import SuperbProblem


class SuperbPR(SuperbProblem):
    @override_parent_cfg(
        corpus=dict(
            _cls=librispeech_for_speech2text,
            dataset_root="???",
        ),
        train_datapipe=dict(
            _cls=Speech2PhonemePipe,
        ),
        train_sampler=dict(
            _cls=MaxTimestampBatchSampler,
            max_timestamp=16000 * 100,
            shuffle=True,
        ),
        valid_datapipe=dict(
            _cls=Speech2PhonemePipe,
        ),
        valid_sampler=dict(
            _cls=FixedBatchSizeBatchSampler,
            batch_size=16,
        ),
        test_datapipe=dict(
            _cls=Speech2PhonemePipe,
        ),
        test_sampler=dict(
            _cls=FixedBatchSizeBatchSampler,
            batch_size=1,
        ),
        downstream=dict(
            _cls=RNNEncoder,
            module="LSTM",
            hidden_size=[1024, 1024],
            dropout=[0.2, 0.2],
            layer_norm=[False, False],
            proj=[False, False],
            sample_rate=[1, 1],
            sample_style="concat",
            bidirectional=True,
        ),
        task=dict(
            _cls=Speech2TextCTCTask,
        ),
    )
    @classmethod
    def setup_problem(cls, **cfg):
        super().setup_problem(**cfg)

    @override_parent_cfg(
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
    @classmethod
    def train(cls, **cfg):
        super().train(**cfg)

    @override_parent_cfg()
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

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
