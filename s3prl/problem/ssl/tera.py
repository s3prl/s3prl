import torch
from torch.nn import L1Loss

from s3prl.corpus.librispeech import librispeech_for_pretrain
from s3prl.dataset.pretrain_tera_pipe import PretrainTeraPipe
from s3prl.nn.transformer_mockingjay import TransformerMockingjay
from s3prl.nn.predictor_mockingjay import PredictorMockingjay
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task import Task
from s3prl.task.feat_reconstruction_task import FeatReconstructionTask
from s3prl.util.configuration import override_parent_cfg
from s3prl.util.workspace import Workspace

from .base import SslProblem

_input_size = 80
_mask_args = dict(
    position_encoding_size=768,  # int, this should be identical to `hidden_size`
    mask_proportion=0.15,  # float, mask this percentage of all spectrogram frames in each sequence at random during MAM training
    mask_consecutive_min=7,  # int, mask this amount of consecutive frames
    mask_consecutive_max=7,  # int, mask this amount of consecutive frames
    mask_allow_overlap=True,  # bool, allow overlap masking
    mask_bucket_ratio=1.5,  # float, only used when overlap is not allowed. sample a mask from each bucket in size of [sampled mask_consecutive * mask_bucket_ratio]
    mask_frequency=0.2,  # float, mask maximum this percentage of frequency bands, set to 0 for no frequency mask
)
_noise_args = dict(
    noise_proportion=0.0,  # float, for this percentage of the time, Gaussian noise will be applied on all frames during MAM training, set to 0 for no noise
)
_audio_config = dict(
    win_ms=25,
    hop_ms=10,
    n_freq=201,
    n_mels=_input_size,
    n_mfcc=13,
    input={
        "channel": 0,
        "cmvn": True,
        "delta": 0,
        "feat_type": "mel",
        "log": True,
    },
    target={
        "channel": 1,
        "cmvn": True,
        "delta": 0,
        "feat_type": "mel",
        "log": True,
    },
)
_pretrain_task_pipe_config = dict(
    _cls=PretrainTeraPipe,
    target_level=-25,
    **_mask_args,
    **_noise_args,
    **_audio_config,
)
_transformer_config = dict(
    hidden_size=768,  # Size of the encoder layers and the pooler layer.
    num_hidden_layers=3,  # Number of hidden layers in the Transformer encoder.
    num_attention_heads=12,  # Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size=3072,  # The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act="gelu",  # The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
    hidden_dropout_prob=0.1,  # The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob=0.1,  # The dropout ratio for the attention probabilities.
    initializer_range=0.02,  # The sttdev of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps=1.0e-12,  # The epsilon used by LayerNorm.
    share_layer=False,  # Share layer weights
    pre_layer_norm=False,  # To apply the pre layer normalization technique introduced in: https://arxiv.org/abs/2002.04745
)


class Tera(SslProblem):
    """
    Tera pre-train problem
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
            _cls=TransformerMockingjay,
            config=_transformer_config,
            input_dim=_input_size,
            output_attentions=False,
            keep_multihead_output=False,
            with_input_module=True,
        ),
        predictor=dict(
            _cls=PredictorMockingjay,
            config=_transformer_config,
            output_dim=_input_size,
            input_dim=None,  # automatically use `hidden_size` from `_transformer_config`
        ),
        task=dict(
            _cls=FeatReconstructionTask,
            loss=L1Loss,
        ),
    )
    @classmethod
    def setup_problem(cls, **cfg):
        """
        This setups the Tera problem, containing train/valid/test datasets & samplers and a task object
        """
        super().setup_problem(**cfg)

    @override_parent_cfg(
        optimizer=dict(
            _cls="torch.optim.AdamW",
            lr=2.0e-4,
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
        all_states = dict(
            Config={},  # placeholder
            SpecHead=task.predictor.state_dict(),
            Transformer=task.upstream.state_dict(),
            Upstream_Config=dict(
                transformer=_transformer_config,
                audio=_audio_config,
                task=dict(sequence_length=0),
            ),
        )
        all_states["Upstream_Config"]["audio"][
            "target_level"
        ] = _pretrain_task_pipe_config["target_level"]
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
