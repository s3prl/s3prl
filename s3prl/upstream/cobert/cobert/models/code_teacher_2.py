# Copyright 2022 ByteDance Inc.
# CoBERT: Self-Supervised Speech Representation Learning Through Code Representation Learning (https://arxiv.org/abs/2210.04062)
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq (https://github.com/facebookresearch/fairseq)

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import register_model, BaseFairseqModel
from fairseq.models.transformer import Embedding
from fairseq.models.wav2vec import Wav2Vec2Config
from fairseq.modules import LayerNorm, PositionalEmbedding, SamePad, TransposeLast, EMAModuleConfig, EMAModule
from fairseq.utils import index_put
from omegaconf import II

from .modules.code_encoder import TransformerEncoder
from ..tasks.cobert_pretraining import CobertPretrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class CodeTeacher2Config(Wav2Vec2Config):
    no_scale_embedding: bool = field(
        default=False,
        metadata={"help": "not scale embedding"},
    )
    no_sin_pos_embed: bool = field(
        default=False,
        metadata={"help": "not sinusoidal positional embedding"},
    )
    learned_pos: bool = field(
        default=False,
        metadata={"help": "whether the sin pos embed is leanred"}
    )
    no_pos_conv: bool = field(
        default=False,
        metadata={"help": "not positional convolution"},
    )
    code_mask: bool = field(
        default=False,
        metadata={"help": "whether to apply mask according to code boundary."
                          "by default, apply span mask."}
    )

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



def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("code_teacher_2", dataclass=CodeTeacher2Config)
class CodeTeacher2(BaseFairseqModel):
    def __init__(self, cfg: CodeTeacher2Config, task_cfg: CobertPretrainingConfig, source_dict):
        super().__init__()

        self.cfg = cfg

        # ema required
        self.ema = None
        self.embed = cfg.encoder_embed_dim

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        # build code embedding
        self.encoder_embed_tokens = self.build_embedding(
            source_dict, self.embed
        )
        self.padding_idx = self.encoder_embed_tokens.padding_idx

        # apply static sin position embedding
        if not cfg.no_sin_pos_embed:
            self.embed_positions = PositionalEmbedding(
                int(task_cfg.max_sample_size / 320) + 1,
                self.embed,
                self.padding_idx,
                learned=cfg.learned_pos,
            )
            logger.info(f"Use sin pos embedding.")
        else:
            self.embed_positions = None
            logger.info(f"Will NOT use sin pos embedding.")

        self.embed_scale = 1.0 if cfg.no_scale_embedding \
            else math.sqrt(self.embed)

        # mask related configs. will not mask by channel
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        # dropout related
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        # use self-defined encoder
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.final_proj = nn.Linear(self.embed, self.embed)

        self.num_updates = 0

        self.code_mask = cfg.code_mask
        logger.info(f"apply_code_mask={self.code_mask}")

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        return Embedding(num_embeddings, embed_dim, padding_idx)

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            if self.encoder.pos_conv is not None:
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
    def build_model(cls, cfg: CodeTeacher2Config, task: CobertPretrainingConfig):
        return cls(cfg, task.cfg, task.source_dictionary)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

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

        return x, mask_indices

    def apply_code_mask(
            self,
            code: torch.Tensor,
            features: torch.Tensor,
            mask_prob: float,
            padding_mask=None
    ):
        """
        Apply mask by code boundary to the feature.
        Args:
            code: the uncompressed source code
            features: the code embedding to be masked
            mask_prob: mask probability, the only parameter to control the mask
            padding_mask: same shape as code. False -> no padding; True -> padding.

        Returns:
            masked_features, mask_indices
        """
        code = code.cpu()
        bsz = code.shape[0]

        all_masks = []

        for i in range(bsz):
            compress_code, counts = torch.unique_consecutive(code[i], return_counts=True)
            if padding_mask is None:
                has_pad = False
            else:
                has_pad = padding_mask[i].any()
            sample_size = (compress_code.shape[0] - 1) if has_pad else compress_code.shape[0]
            mask_num = int(sample_size * mask_prob + np.random.rand())

            mask_idx = np.random.choice(sample_size, size=mask_num, replace=False)
            mask_idx = np.sort(mask_idx)

            sample_mask = np.full(shape=(compress_code.shape[0],), fill_value=False)
            sample_mask[mask_idx] = True

            sample_mask = np.reshape(sample_mask, newshape=(-1, 1))

            uncompress_sample_mask = np.repeat(sample_mask, counts)

            all_masks.append(uncompress_sample_mask)

        mask_indices = torch.from_numpy(np.stack(all_masks)).to(features.device)
        features = index_put(features, mask_indices, self.mask_emb)
        return features, mask_indices

    def forward_embedding(self, src_tokens):
        token_embedding = self.encoder_embed_tokens(src_tokens)
        x = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        return x

    def forward(
            self,
            source_codes,
            source=None,
            padding_mask=None,
            mask=True,
            features_only=False,
            layer=None,
            mask_indices=None,
            mask_channel_indices=None,
            padding_count=None,
    ):
        # will not be used here
        source = None
        # code -> embedding
        encoder_padding_mask = source_codes.eq(self.padding_idx)
        has_pads = encoder_padding_mask.any()
        # B,T,C
        features = self.forward_embedding(src_tokens=source_codes)

        if has_pads:
            features = features * (1 - encoder_padding_mask.unsqueeze(-1).type_as(features))
        # B * T * C
        features = self.layer_norm(features)

        # compare code index with pad index to get padding mask
        if encoder_padding_mask is not None and encoder_padding_mask.any():
            padding_mask = encoder_padding_mask
        else:
            padding_mask = None

        pre_encoder_features = features.clone()
        features = self.dropout_input(features)

        if mask:
            if self.code_mask:
                x, mask_indices = self.apply_code_mask(
                    source_codes,
                    features,
                    self.mask_prob,
                    padding_mask
                )
            else:
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
            self.ema.model.eval()

            # the only option is ema transformer
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

            # the rest are copied from super class
            target_layer_results = [l[2] for l in y["layer_results"]]

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

            y = y[mask_indices]

        x = x[mask_indices]
        x = self.final_proj(x)

        sz = x.size(-1)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        result["losses"]["regression"] = loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(x.float())

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

        # add log for mask token number
        result["total_token_num"] = torch.sum((~padding_mask).int()) if padding_mask is not None \
                                    else source_codes.numel()

        return result

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
            source_codes=source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )
