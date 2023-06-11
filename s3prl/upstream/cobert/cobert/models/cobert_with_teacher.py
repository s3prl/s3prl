# Copyright 2022 ByteDance Inc.
# CoBERT: Self-Supervised Speech Representation Learning Through Code Representation Learning (https://arxiv.org/abs/2210.04062)
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq (https://github.com/facebookresearch/fairseq)

import logging
import math
import os

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING, II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq import checkpoint_utils, tasks
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)
from fairseq.modules import (
    GradMultiply,
    LayerNorm,
)
from fairseq.utils import index_put

from .code_teacher_1 import CodeTeacher1
from .code_teacher_2 import CodeTeacher2
from .hubert_teacher import HubertTeacherModel

logger = logging.getLogger(__name__)

LENGTH_TOLERANCE = 4


@dataclass
class CobertWithTeacherConfig(Wav2Vec2Config):
    code_teacher_ckpt: str = field(
        default=MISSING,
        metadata={"help": "The path to the ckpt of the teacher model. "
                          "If not provides, will act the same as origin data2vec."}
    )
    code_teacher_type: str = field(
        default="code_teacher_1",
        metadata={"help": "the type of code teacher."
                          "optionally, a speech encoder like HuBERT or data2vec_audio"
                          "can also be a teacher."}
    )
    code_teacher_min_layer: int = field(
        default=0,
        metadata={"help": "inclusive. The first layer index whose output counts as teacher."}
    )
    code_teacher_max_layer: int = field(
        default=-1,
        metadata={"help": "exclusive. The last layer index whose output counts as teacher."}
    )
    multi_outputs: bool = field(
        default=False,
        metadata={"help": "whether to compute loss using multiplie output layers."}
    )
    code_loss_only: bool = field(
        default=False,
        metadata={"help": "whether to compute only the loss from code teacher. "
                          "if true, ema model will not be used and the training is faster."
                          "for ema loss only, use data2vec_audio directly."}
    )
    # refer to the dir defined in SpeechCodePretrainingConfig
    normalize: bool = II("task.normalize")
    code_path: str = II("task.data")

    # original Data2VecAudioConfig
    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )


def _load_code_teacher_1(_cfg: CobertWithTeacherConfig) -> CodeTeacher1:
    # should not override args, we need to keep the teacher as it was.
    state = checkpoint_utils.load_checkpoint_to_cpu(_cfg.code_teacher_ckpt)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])

    # no need to check normalize, because code hubert does not require normalize
    # assert _cfg.normalize == w2v_args.task.normalize

    w2v_args.task.data = _cfg.code_path
    pretrain_task = tasks.setup_task(w2v_args.task)
    # This will load the stored "dictionaries" object
    pretrain_task.load_state_dict(state["task_state"])

    model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
    # can be strict here
    model.load_state_dict(state["model"], strict=True)

    # model.remove_pretraining_modules()
    return model


def _load_data2vec_audio(_cfg: CobertWithTeacherConfig):
    # This loads both data2vec_audio model and data2vec_audio_code model.
    state = checkpoint_utils.load_checkpoint_to_cpu(_cfg.code_teacher_ckpt)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])

    # have to check normalize. the teacher use the same audio as the student here.
    assert _cfg.normalize == w2v_args.task.normalize

    pretrain_task = tasks.setup_task(w2v_args.task)
    # This will load the stored "dictionaries" object
    pretrain_task.load_state_dict(state["task_state"])

    # prevent loading from wrong ckpt path
    if "code_teacher_ckpt" in w2v_args.model:
        logger.info(f"Replaced code_teacher_ckpt {w2v_args.model['code_teacher_ckpt']}...")
        w2v_args.model["code_teacher_ckpt"] = "placeholder"

    # does not need the code_teacher_proj for feature extraction
    if "multi_outputs" in w2v_args.model:
        logger.info(f"Set multi_outputs {w2v_args.model['multi_outputs']} to false")
        w2v_args.model["multi_outputs"] = False
    model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
    # delete possible teacher models (code & ema) since not needed.
    # also because we do not provide code teacher here.
    for k in list(state["model"].keys()):
        if "code_teacher" in k or "_ema" in k:
            del state["model"][k]
    # can be strict here
    model.load_state_dict(state["model"], strict=True)

    model.remove_pretraining_modules()
    return model


def _load_code_teacher_2(_cfg: CobertWithTeacherConfig) -> CodeTeacher2:
    state = checkpoint_utils.load_checkpoint_to_cpu(_cfg.code_teacher_ckpt)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])

    # no need to check normalization since data2vec_code does not need audio input
    # assert _cfg.normalize == w2v_args.task.normalize

    # NOTE: be sure to put dict.km.txt at /opt/tiger/pretrain_meta
    pretrain_task = tasks.setup_task(w2v_args.task)
    pretrain_task.load_state_dict(state["task_state"])

    model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
    # delete possible teacher models (ema) since not needed.
    for k in list(state["model"].keys()):
        if "_ema" in k:
            del state["model"][k]
    # can be strict here
    model.load_state_dict(state["model"], strict=True)

    model.remove_pretraining_modules()
    return model


def _load_hubert(_cfg: CobertWithTeacherConfig) -> HubertTeacherModel:
    # do code input inference here.
    state = checkpoint_utils.load_checkpoint_to_cpu(_cfg.code_teacher_ckpt)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])

    # need to check normalize. The performance should be better if the normalization matches
    assert _cfg.normalize == w2v_args.task.normalize

    # do not need this. we provide data using Data2VecAudioCodeModel.
    # w2v_args.task.data = _cfg.code_path
    pretrain_task = tasks.setup_task(w2v_args.task)
    # This will load the stored "dictionaries" object
    pretrain_task.load_state_dict(state["task_state"])
    # use hubert_teacher model instead of original hubert model
    w2v_args.model['_name'] = "hubert_teacher"
    model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
    # can be strict here
    model.load_state_dict(state["model"], strict=True)

    model.remove_pretraining_modules()
    return model


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("cobert_with_teacher", dataclass=CobertWithTeacherConfig)
class CobertWithTeacherModel(BaseFairseqModel):
    cfg: CobertWithTeacherConfig
    code_teacher_choices = ["code_teacher_1", "code_teacher_2", "data2vec_audio", "hubert"]

    def __init__(self, cfg: CobertWithTeacherConfig):
        super().__init__()
        # original data2vec_audio model
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.extractor_embed = feature_enc_layers[-1][0]

        self.ema = None
        self.embed = cfg.encoder_embed_dim

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = nn.Linear(self.extractor_embed, cfg.encoder_embed_dim)

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.extractor_embed)

        self.final_proj = nn.Linear(self.embed, self.embed)

        self.num_updates = 0

        # load another teacher model
        self.code_teacher_model = None
        self.code_teacher_type = cfg.code_teacher_type
        if cfg.code_teacher_ckpt is not None and os.path.exists(cfg.code_teacher_ckpt):
            logger.info(f"Will load code teacher {self.code_teacher_type} from {cfg.code_teacher_ckpt}")
            assert self.code_teacher_type in CobertWithTeacherModel.code_teacher_choices
            if self.code_teacher_type == "code_teacher_1":
                self.code_teacher_model: CodeTeacher1 = _load_code_teacher_1(cfg)
            elif self.code_teacher_type == "code_teacher_2":
                self.code_teacher_model: CodeTeacher2 = _load_code_teacher_2(cfg)
            elif self.code_teacher_type == "data2vec_audio":
                self.code_teacher_model = _load_data2vec_audio(cfg)
            elif self.code_teacher_type == "hubert":
                self.code_teacher_model: HubertTeacherModel = _load_hubert(cfg)
            self.code_teacher_model.requires_grad_(requires_grad=False)
            self.code_teacher_model.eval()
            # log the parameters to make sure all parameters are correctly set
            for name, param in self.named_parameters():
                logger.debug(f"{name}.requires_grad={param.requires_grad}")
        else:
            logger.warning(f"Connot load code teacher from {cfg.code_teacher_ckpt}. Make sure this is fine-tuning.")

        self.multi_outputs = cfg.multi_outputs
        logger.info(f"multi-outputs={self.multi_outputs}")
        if self.multi_outputs:
            self.code_teacher_proj = nn.Linear(self.embed, self.embed)

        self.code_loss_only = cfg.code_loss_only
        logger.info(f"code_loss_only={self.code_loss_only}")

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.final_proj is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: CobertWithTeacherConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def apply_mask(
            self,
            x,
            padding_mask,
            mask_indices=None,
            mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
            self,
            source,
            source_codes=None,
            source_codes_lengths=None,
            padding_mask=None,
            mask=True,
            features_only=False,
            layer=None,
            mask_indices=None,
            mask_channel_indices=None,
            padding_count=None,
    ):
        features = source

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        features = features.transpose(1, 2)

        features = self.layer_norm(features)

        orig_padding_mask = padding_mask

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        pre_encoder_features = None
        if self.cfg.ema_transformer_only:
            pre_encoder_features = features.clone()

        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
        else:
            x = features
            mask_indices = None

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=layer,
        )

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

        result = {
            "losses": {},
        }

        with torch.no_grad():
            if self.code_loss_only:
                y = None
            else:
                self.ema.model.eval()

                if self.cfg.ema_transformer_only:
                    y, layer_results = self.ema.model.extract_features(
                        pre_encoder_features,
                        padding_mask=padding_mask,
                        min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
                    )
                    y = {
                        "x": y,
                        "padding_mask": padding_mask,
                        "layer_results": layer_results,
                    }
                else:
                    y = self.ema.model.extract_features(
                        source=source,
                        padding_mask=orig_padding_mask,
                        mask=False,
                    )
                # TBC
                target_layer_results = [l[2] for l in y["layer_results"]]
                y = self._aggregate_features(target_layer_results)

            # compute code teacher representation
            # T x B x C
            code_y = None
            if self.code_teacher_type == "code_teacher_1":
                code_y, feat_padding_mask = self._get_code_teacher_1_feature(source_codes, orig_padding_mask)
            if self.code_teacher_type == "code_teacher_2":
                code_y = self._get_code_teacher_2_feature(source_codes)
            if self.code_teacher_type == "data2vec_audio":
                code_y = self._get_data2vec_audio_feature(source, orig_padding_mask)
            if self.code_teacher_type == "hubert":
                code_y = self._get_hubert_feature(source, orig_padding_mask)
            assert code_y is not None and \
                   len(code_y) == self.cfg.code_teacher_max_layer - self.cfg.code_teacher_min_layer

            # T x B x C -> B x T x C
            code_y = self._aggregate_features(code_y)

            # possible trims here
            # do not trim for audio input teacher
            if self.code_teacher_type != "data2vec_audio":
                # trim feature to mask sure the dimensions match
                code_len = code_y.size(1)
                feature_len = mask_indices.size(1)
                if code_len < feature_len:
                    mask_indices = mask_indices[:, :code_len]
                    if abs(code_len - feature_len) > LENGTH_TOLERANCE:
                        logger.info(f"{code_len} < {feature_len}, trim y & mask")
                        logger.info(source_codes.size())
                        logger.info(mask_indices.size())

                    if not self.code_loss_only:
                        y = y[:, :code_len, :]
                        if abs(code_len - feature_len) > LENGTH_TOLERANCE:
                            logger.info(y.size())
                # trim code to match the dim of feature
                if code_len > feature_len:
                    code_y = code_y[:, :feature_len, :]
                    if abs(code_len - feature_len) > LENGTH_TOLERANCE:
                        logger.info(f"{code_len} > {feature_len}, trim code")
                        logger.info(code_y.size())

            if not self.code_loss_only:
                y = y[mask_indices]
            code_y = code_y[mask_indices]

        # trim x outside torch.no_grad(), just in case the gradient over x will be ignored
        if self.code_teacher_type != "data2vec_audio" and code_len < feature_len:
            x = x[:, :code_len, :]
            if abs(code_len - feature_len) > LENGTH_TOLERANCE:
                logger.info(f"{code_len} < {feature_len}, trim x")
                logger.info(x.size())
        x = x[mask_indices]

        if self.multi_outputs:
            # compute another projection for code teacher
            x_for_code_teacher = self.code_teacher_proj(x)

        x = self.final_proj(x)

        sz = x.size(-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        if not self.code_loss_only:
            ema_loss = self._compute_loss(pred=x, target=y)
            result["losses"]["regression"] = ema_loss.sum() * scale

        if self.multi_outputs:
            code_loss = self._compute_loss(pred=x_for_code_teacher, target=code_y)
        else:
            code_loss = self._compute_loss(pred=x, target=code_y)

        result["losses"]["code_teacher"] = code_loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = code_loss.numel()

        with torch.no_grad():
            if not self.code_loss_only:
                result["target_var"] = self.compute_var(y)
                result["target_mean"] = self.compute_mean(y)

            result["pred_var"] = self.compute_var(x.float())
            result["code_teacher_var"] = self.compute_var(code_y)

            result["pred_mean"] = self.compute_mean(x.float())
            result["code_teacher_mean"] = self.compute_mean(code_y)

            if self.multi_outputs:
                result["pred_for_code_teacher_var"] = self.compute_var(x_for_code_teacher.float())

        if not self.code_loss_only:
            if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
                logger.error(
                    f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
                )
                raise Exception(
                    f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
                )
        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    def _aggregate_features(self, target_layer_results: List[torch.FloatTensor]):
        """
        Normalize and aggregate multiple layers' outputs.
        Args:
            target_layer_results: outputs from multiple layers.

        Returns:
            a tensor
        """
        permuted = False
        if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
            target_layer_results = [
                tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
            ]
            permuted = True

        if self.cfg.batch_norm_target_layer:
            target_layer_results = [
                F.batch_norm(
                    tl.float(), running_mean=None, running_var=None, training=True
                )
                for tl in target_layer_results
            ]

        if self.cfg.instance_norm_target_layer:
            target_layer_results = [
                F.instance_norm(tl.float()) for tl in target_layer_results
            ]

        if permuted:
            target_layer_results = [
                tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
            ]

        if self.cfg.group_norm_target_layer:
            target_layer_results = [
                F.layer_norm(tl.float(), tl.shape[-2:])
                for tl in target_layer_results
            ]

        if self.cfg.layer_norm_target_layer:
            target_layer_results = [
                F.layer_norm(tl.float(), tl.shape[-1:])
                for tl in target_layer_results
            ]

        y = sum(target_layer_results) / len(target_layer_results)

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y.float(), y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

        if not permuted:
            y = y.transpose(0, 1)

        return y

    def _get_code_teacher_1_feature(self, source_codes, padding_mask):
        # use .eval() every time
        self.code_teacher_model: CodeTeacher1
        self.code_teacher_model.eval()
        # T x B x C
        all_layer_results, feat_padding_mask = self.code_teacher_model.extract_features(
            source=source_codes,
            padding_mask=padding_mask,
            mask=False,
            output_layer=None,
            return_all_layers=True
        )
        all_features = [layer_result[2] for layer_result in all_layer_results]
        return all_features[self.cfg.code_teacher_min_layer:self.cfg.code_teacher_max_layer], feat_padding_mask

    def _get_data2vec_audio_feature(self, source, padding_mask):
        self.code_teacher_model.eval()
        ret = self.code_teacher_model.extract_features(
            source=source,
            padding_mask=padding_mask,
            mask=False,
        )
        all_layer_results = ret["layer_results"]
        all_features = [layer_result[2] for layer_result in all_layer_results]
        return all_features[self.cfg.code_teacher_min_layer:self.cfg.code_teacher_max_layer]

    def _get_code_teacher_2_feature(self, source):
        self.code_teacher_model: CodeTeacher2
        self.code_teacher_model.eval()
        ret = self.code_teacher_model.extract_features(
            source,
            padding_mask=None,
            mask=False
        )
        all_layer_results = ret["layer_results"]
        all_features = [layer_result[2] for layer_result in all_layer_results]
        return all_features[self.cfg.code_teacher_min_layer:self.cfg.code_teacher_max_layer]

    def _get_hubert_feature(self, source, padding_mask):
        # use .eval() every time
        self.code_teacher_model: HubertTeacherModel
        self.code_teacher_model.eval()
        # T x B x C
        all_layer_results, _ = self.code_teacher_model.extract_features(
            source=source,
            padding_mask=padding_mask,
            mask=False,
            output_layer=None,
            return_all_layers=True
        )
        all_features = [layer_result[2] for layer_result in all_layer_results]
        return all_features[self.cfg.code_teacher_min_layer:self.cfg.code_teacher_max_layer]

    def _compute_loss(self, pred, target):
        if self.loss_beta == 0:
            loss = F.mse_loss(pred.float(), target.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                pred.float(), target.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)
        return loss

    @staticmethod
    def compute_mean(y):
        return y.mean()

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
            self, source, padding_mask, mask=False, layer=None
    ):
        res = self.forward(
            source=source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        # original data2vec_audio removal
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )
        # code teacher removal
        self.code_teacher_model = None
        if self.multi_outputs:
            self.code_teacher_proj = None
