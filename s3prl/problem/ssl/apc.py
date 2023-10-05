import torch
from torch.nn import L1Loss

from s3prl.corpus.librispeech import librispeech_for_pretrain
from s3prl.dataset.pretrain_apc_pipe import PretrainApcPipe
from s3prl.nn.predictor_identity import PredictorIdentity
from s3prl.nn.rnn_apc import RnnApc
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task import Task
from s3prl.task.autoregressive_reconstruction_task import (
    AutoregressiveReconstructionTask,
)
from s3prl.util.configuration import override_parent_cfg
from s3prl.util.workspace import Workspace

from .base import SslProblem

_input_size = 80
_audio_config = dict(
    feat_type="fbank",  # Feature type
    feat_dim=_input_size,  # Feature dimension
    frame_length=25,  # Window size in ms
    frame_shift=10,  # Hop size in ms
    decode_wav=False,
    cmvn=True,  # Apply uttr.-wised CMVN on Mel spectrogram
)
_pretrain_task_pipe_config = dict(
    _cls=PretrainApcPipe,
    n_future=5,
    n_jobs=8,
    **_audio_config,
)


class Apc(SslProblem):
    """
    Apc pre-train problem
    """

    @override_parent_cfg(
        corpus=dict(
            _cls=librispeech_for_pretrain,
            dataset_root="???",
        ),
        train_datapipe=_pretrain_task_pipe_config,
        train_sampler=dict(
            _cls=MaxTimestampBatchSampler,
            max_timestamp=16000 * 20,
            shuffle=True,
        ),
        valid_datapipe=_pretrain_task_pipe_config,
        valid_sampler=dict(
            _cls=FixedBatchSizeBatchSampler,
            batch_size=2,
        ),
        test_datapipe=_pretrain_task_pipe_config,
        test_sampler=dict(
            _cls=FixedBatchSizeBatchSampler,
            batch_size=2,
        ),
        upstream=dict(
            _cls=RnnApc,
            input_size=_input_size,
            num_layers=3,
            hidden_size=512,
            dropout=0.1,
            residual=True,
        ),
        predictor=dict(
            _cls=PredictorIdentity,
        ),
        task=dict(
            _cls=AutoregressiveReconstructionTask,
            loss=L1Loss,
        ),
    )
    @classmethod
    def setup_problem(cls, **cfg):
        """
        This setups the Apc problem, containing train/valid/test datasets & samplers and a task object
        """
        super().setup_problem(**cfg)

    @override_parent_cfg(
        optimizer=dict(
            _cls="torch.optim.AdamW",
            lr=0.0001,  # set to 0.00001 for some datasets if you encounter NaN during training
        ),
        trainer=dict(
            total_steps=1000000,
            eval_step=50000,
            save_step=50000,
            gradient_clipping=5.0,
            gradient_accumulate_steps=4,
            valid_metric="loss",
            valid_higher_better=False,
        ),
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

    @override_parent_cfg()
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @classmethod
    def save_additional(
        cls,
        additional_dir: Workspace,
        workspace: Workspace,
        task: Task,
    ):
        setup_problem_cfg = workspace.get_cfg(cls.setup_problem)
        setup_problem_cfg["upstream"].pop("_cls")
        setup_problem_cfg["upstream"].pop("input_size")
        apc_config = dict(
            model=dict(
                paras=setup_problem_cfg["upstream"],
            ),
            data=dict(
                audio=_audio_config,
            ),
        )
        all_states = dict(
            config=apc_config,
            model=task.upstream.state_dict(),
            Upstream_Config=apc_config,
        )
        torch.save(
            all_states, str(additional_dir.parent.resolve()) + "/all_states.ckpt"
        )

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
