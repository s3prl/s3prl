# --------------------------------------------------------
# LightHuBERT: Lightweight and Configurable Speech Representation Learning with Once-for-All Hidden-Unit BERT (https://arxiv.org/pdf/2203.15610.pdf)
# Github source: https://github.com/mechanicalsea/lighthubert
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional.fairseq_utils import get_activation_fn, index_put, pad_to_multiple
from .fairseq_modules import SamePad, TransposeLast
from .scaling_conv import SConv1d
from .scaling_layernorm import SLayerNorm
from .scaling_linear import SLinear
from .scaling_multihead import SMHA


class STransformerSentenceEncoderLayer(nn.Module):
    """TransformerSentenceEncoderLayer: variable input (i.e., output) size, attention heads, and ffn embedding.
    The `fc1` and `fc2` enable sharing weights with FFN ratio, e.g., 2.5, 3.0, 3.5, 4.0

    Dynamic: self.self_attn, self.fc1, self.fc2, self.self_attn_layer_norm, self.final_layer_norm

    wav2vec2:TransformerSentenceEncoderLayer(
        embedding_dim=self.embedding_dim,
        ffn_embedding_dim=args.encoder_ffn_embed_dim,
        num_attention_heads=args.encoder_attention_heads,
        dropout=self.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_fn=args.activation_fn,
        layer_norm_first=args.layer_norm_first,
    )
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = SMHA(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )  # lighthubert component

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = SLayerNorm(
            self.embedding_dim
        )  # lighthubert component
        self.fc1 = SLinear(
            self.embedding_dim,
            ffn_embedding_dim,
            in_splits=1,
            out_splits=ffn_embedding_dim // self.embedding_dim,
        )  # lighthubert component
        self.fc2 = SLinear(
            ffn_embedding_dim,
            self.embedding_dim,
            in_splits=ffn_embedding_dim // self.embedding_dim,
            out_splits=1,
        )  # lighthubert component

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = SLayerNorm(self.embedding_dim)  # lighthubert component

        # scaling module
        self.sample_atten_dim = None
        self.sample_embed_dim = None
        self.sample_ffn_embed = None
        self.sample_heads_num = None
        self.sample_attn_swz = "global"
        # replace self_attn, self_attn_layer_norm, fc1, fc2, final_layer_norm

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)

    def set_sample_config(
        self,
        sample_atten_dim: int,
        sample_embed_dim: int,
        sample_ffn_embed: int,
        sample_heads_num: int,
        sample_sliding_attn_window="global",
    ):
        assert (
            sample_atten_dim == sample_heads_num * 64
        ), f"{sample_atten_dim}-{sample_heads_num}"
        self.sample_atten_dim = sample_atten_dim
        self.sample_embed_dim = sample_embed_dim
        self.sample_ffn_embed = sample_ffn_embed
        self.sample_heads_num = sample_heads_num
        self.sample_attn_swz = sample_sliding_attn_window
        self._sample_parameters()

    def _sample_parameters(self):
        self.self_attn.set_sample_config(
            self.sample_atten_dim,
            self.sample_heads_num,
            self.sample_embed_dim,
            self.sample_attn_swz,
        )
        self.self_attn_layer_norm.set_sample_config(self.sample_embed_dim)
        self.fc1.set_sample_config(
            self.sample_embed_dim,
            self.sample_ffn_embed,
            1,
            self.sample_ffn_embed / self.sample_embed_dim,
        )
        self.fc2.set_sample_config(
            self.sample_ffn_embed,
            self.sample_embed_dim,
            self.sample_ffn_embed / self.sample_embed_dim,
            1,
        )
        self.final_layer_norm.set_sample_config(self.sample_embed_dim)

    @property
    def sample_modules(self):
        return {
            "self_attn": SMHA,
            "self_attn_layer_norm": SLayerNorm,
            "fc1": SLinear,
            "fc2": SLinear,
            "final_layer_norm": SLayerNorm,
        }

    def calc_sampled_param_num(self):
        total_params = 0
        for module_name in self.sample_modules.keys():
            m = getattr(self, module_name)
            total_params += m.calc_sampled_param_num()
        return total_params

    def get_complexity(self, sequence_length):
        total_flops = 0
        for module_name in self.sample_modules.keys():
            m = getattr(self, module_name)
            total_flops += m.get_complexity(sequence_length)
        # 2 residual connection
        total_flops += 2 * sequence_length * self.sample_embed_dim
        # activation
        total_flops += sequence_length * self.sample_ffn_embed
        return total_flops


def make_sconv_pos(e, k, g):
    pos_conv = SConv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv


def init_sbert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of scaling_linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        1. If normal_init_proj_weights is set then weights of
           in_project_weight for scaling_MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, SLinear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, SMHA):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class STransformerEncoder(nn.Module):
    """TransformerEncoder: variable layers number, input dim (or outout),
    heads numbers, and ffn embedding dim.

    Note that `input dim (or outout)` now is fixed to be subject
    to hidden representations of teacher model.

    Dynamic: self.layers, self.pos_conv.0, self.layer_norm

    wav2vec2:TransformerEncoder(args)
    """

    def build_sencoder_layer(self, args):
        assert args.layer_type == "transformer", args.layer_type
        if args.layer_type == "transformer":
            layer = STransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
        return layer

    def __init__(self, args):

        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_sconv_block(e, k, g, l):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            SConv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            SLayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_sconv_block(
                self.embedding_dim, k, args.conv_pos_groups, num_layers
            )

        else:
            self.pos_conv = make_sconv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
            )

        self.layers = nn.ModuleList(
            [self.build_sencoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = SLayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_sbert_params)

        self.args = args
        assert args.encoder_layers in [12], f"encoder layers: {args.encoder_layers}"
        # layer block: 1-4, 5-8, 9-12
        self.depth_maps = {
            6: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            7: [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            8: [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
            9: [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            10: [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            11: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            12: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }  # 1: keep layer, 0: drop layer
        self.sample_layer_num = args.encoder_layers
        self.sample_atten_dim = None
        self.sample_embed_dim = None
        self.sample_ffn_embed = None
        self.sample_heads_num = None
        self.sample_attn_swz = None

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
    ):
        """Refactor it to enable varible layers number"""
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        layerdrop_index = self.depth_maps[self.sample_layer_num]
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if layerdrop_index[i] == 1 and (
                not self.training or dropout_probability > self.layerdrop
            ):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length],
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def set_sample_config(
        self,
        sample_layer_num: int,
        sample_atten_dim: List[int],
        sample_embed_dim: int,
        sample_ffn_embed: List[int],
        sample_heads_num: List[int],
        sample_attn_swz: List,
    ):
        self.sample_layer_num = sample_layer_num
        self.sample_atten_dim = sample_atten_dim
        self.sample_embed_dim = sample_embed_dim
        self.sample_ffn_embed = sample_ffn_embed
        self.sample_heads_num = sample_heads_num
        self.sample_attn_swz = sample_attn_swz
        self._sample_parameters()

    def _sample_parameters(self):
        if getattr(self.args, "pos_conv_depth", 1) > 1:
            pos_conv_depth = getattr(self.args, "pos_conv_depth", 1)
            prune_encoder_pos_conv = getattr(self.args, "prune_encoder_pos_conv", True)
            sample_input_size = (
                self.sample_embed_dim
                if prune_encoder_pos_conv
                else self.args.encoder_embed_dim
            )
            for i, mi in enumerate(self.pos_conv):
                if i == 0:
                    mi[0].set_sample_config(self.sample_embed_dim, sample_input_size)
                    mi[3].set_sample_config(sample_input_size)
                elif i < pos_conv_depth - 1:
                    mi[0].set_sample_config(sample_input_size, sample_input_size)
                    mi[3].set_sample_config(sample_input_size)
                else:
                    mi[0].set_sample_config(sample_input_size, self.sample_embed_dim)
                    mi[3].set_sample_config(self.sample_embed_dim)
        else:
            self.pos_conv[0].set_sample_config(
                self.sample_embed_dim,
                self.sample_embed_dim,
            )
        layerdrop_index = self.depth_maps[self.sample_layer_num]
        for i, layer in enumerate(self.layers):
            if layerdrop_index[i] == 1:
                # enable layer-drop training
                layer_i = sum(layerdrop_index[: i + 1]) - 1
                layer.set_sample_config(
                    self.sample_atten_dim[layer_i],
                    self.sample_embed_dim,
                    self.sample_ffn_embed[layer_i],
                    self.sample_heads_num[layer_i],
                    self.sample_attn_swz[layer_i],
                )
        self.layer_norm.set_sample_config(self.sample_embed_dim)

    def calc_sampled_param_num(self):
        total_params = 0
        layerdrop_index = self.depth_maps[self.sample_layer_num]
        total_params += sum(
            layer.calc_sampled_param_num()
            for i, layer in enumerate(self.layers)
            if layerdrop_index[i] == 1
        )
        if getattr(self.args, "pos_conv_depth", 1) > 1:
            for mi in self.pos_conv:
                total_params += mi[0].calc_sampled_param_num()
        else:
            total_params += self.pos_conv[0].calc_sampled_param_num()
        total_params += self.layer_norm.calc_sampled_param_num()
        return total_params

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sum(
            layer.get_complexity(sequence_length) for layer in self.layers
        )
        # pos_conv
        if getattr(self.args, "pos_conv_depth", 1) > 1:
            for mi in self.pos_conv:
                total_flops += (
                    self.sample_embed_dim
                    * sequence_length
                    * (
                        self.sample_embed_dim / mi[0].groups * mi[0].kernel_size
                        + int(mi[0].bias is not None)
                    )
                )  # conv
                total_flops += self.sample_embed_dim * sequence_length  # layernorm
        else:
            total_flops += (
                self.sample_embed_dim
                * sequence_length
                * (
                    self.sample_embed_dim
                    / self.pos_conv[0].groups
                    * self.pos_conv[0].kernel_size
                    + int(self.pos_conv[0].bias is not None)
                )
            )
        # layer_norm
        total_flops += self.sample_embed_dim * sequence_length
        return total_flops
