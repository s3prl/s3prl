# --------------------------------------------------------
# LightHuBERT: Lightweight and Configurable Speech Representation Learning with Once-for-All Hidden-Unit BERT (https://arxiv.org/pdf/2203.15610.pdf)
# Github source: https://github.com/mechanicalsea/lighthubert
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..functional.sliding_attn import (
    global_attention_forward,
    slide_window_attention_forward,
)
from .fairseq_modules import quant_noise
from .scaling_linear import SLinear


class SMHA(nn.Module):
    """SMHA (Scaling MultiheadAttention): variable input (i.e., output) size and heads number.
    where in_embed_dim = out_embed_dim, qkv_embed_dim = 64 * num_heads

    wav2vec2:MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=...,
        self_attention=True,
    )
    Module: self.k_proj, self.v_proj, self.q_proj, self.out_proj
    None: self.bias_k, self.bias_v

    __base__: fairseq.modules.MultiheadAttention
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,  # if 0, no quant_noise
        qn_block_size=8,
        sliding_attn_window="global",
        slide_mode="stride",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            SLinear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )  # lighthubert component
        self.v_proj = quant_noise(
            SLinear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )  # lighthubert component
        self.q_proj = quant_noise(
            SLinear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )  # lighthubert component

        self.out_proj = quant_noise(
            SLinear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )  # lighthubert component

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        # scaling attention
        self.slide_mode = slide_mode
        self.sliding_attn_window = sliding_attn_window

        self.sample_qkv_embed_dim = None
        self.sample_num_heads = None
        self.sample_in_embed_dim = None
        self.sample_attn_swz = "global"

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def set_slide_mode(self, slide_mode):
        assert slide_mode in ["stride", "mask"]
        self.slide_mode = slide_mode

    def set_sample_config(
        self,
        sample_qkv_embed_dim: int,
        sample_num_heads: int,
        sample_in_embed_dim: int,
        sample_attn_swz="global",
    ):
        if sample_qkv_embed_dim is None:
            sample_qkv_embed_dim = self.embed_dim
        if sample_num_heads is None:
            sample_num_heads = self.num_heads
        if sample_in_embed_dim is None:
            sample_in_embed_dim = self.embed_dim
        if sample_attn_swz is None or sample_attn_swz == "global":
            sample_attn_swz = "global"
        assert sample_num_heads * 64 == sample_qkv_embed_dim, ValueError(
            f"heads num {sample_num_heads} * 64 != qkv dim {sample_qkv_embed_dim}"
        )
        self.sample_qkv_embed_dim = sample_qkv_embed_dim
        self.sample_num_heads = sample_num_heads
        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_attn_swz = sample_attn_swz
        self._sample_parameters()

    def _sample_parameters(self):
        self.k_proj.set_sample_config(
            self.sample_in_embed_dim, self.sample_qkv_embed_dim
        )
        self.v_proj.set_sample_config(
            self.sample_in_embed_dim, self.sample_qkv_embed_dim
        )
        self.q_proj.set_sample_config(
            self.sample_in_embed_dim, self.sample_qkv_embed_dim
        )
        self.out_proj.set_sample_config(
            self.sample_qkv_embed_dim, self.sample_in_embed_dim
        )

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Implement via fairseq multihead_attention with longformer's sliding attention window

        Args:
            query, key, value: (seq_len, batch_size, hidden_dim)

            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
                binary ByteTensor of shape `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask: 2D or 3D mask that prevents attention to certain positions.
                A 2D mask will be broadcasted for all the batches while a 3D mask
                allows to specify a different mask for the entries of each batch.
                When the value is 1, the corresponding value on the attention
                layer will be added with -1e4 (float16) or -1e8 (float32) or -1e2 (float8).
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the embedding for `tgt_i`,
                we exclude (mask out) `src_j`. This is useful for strided self-attention.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if self.sample_in_embed_dim is None:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"
        else:
            assert (
                embed_dim == self.sample_in_embed_dim
            ), f"query dim {embed_dim} != {self.sample_in_embed_dim}"
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        # incremental_state is None
        assert incremental_state is None
        saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k[:, :, : k.size(-1)].repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v[:, :, : v.size(-1)].repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        sample_num_heads = (
            self.num_heads if self.sample_num_heads is None else self.sample_num_heads
        )
        q = (
            q.contiguous()
            .view(tgt_len, bsz * sample_num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * sample_num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * sample_num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        if self.sample_attn_swz == "global":
            attn, unnormalized_attn_weights = global_attention_forward(
                q,
                k,
                v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                num_heads=sample_num_heads,
                dropout_p=self.dropout_module.p,
                training=self.training,
            )
        else:
            attn, unnormalized_attn_weights = slide_window_attention_forward(
                q,
                k,
                v,
                self.sample_attn_swz,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                num_heads=sample_num_heads,
                dropout_p=self.dropout_module.p,
                training=self.training,
                mode=self.slide_mode,
            )

        assert list(attn.size()) == [bsz * sample_num_heads, tgt_len, self.head_dim]

        attn_embed_dim = embed_dim
        if self.sample_qkv_embed_dim is not None:
            attn_embed_dim = self.sample_qkv_embed_dim
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, attn_embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, attn_embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights and isinstance(unnormalized_attn_weights, Tensor):
            wsz = src_len
            if self.slide_mode == "stride" and not self.sample_attn_swz == "global":
                wsz = self.sample_attn_swz + 1
            attn_weights = unnormalized_attn_weights.view(
                bsz, sample_num_heads, tgt_len, wsz
            )
        return attn, attn_weights

    def calc_sampled_param_num(self):
        total_params = 0
        total_params += self.k_proj.calc_sampled_param_num()
        total_params += self.v_proj.calc_sampled_param_num()
        total_params += self.q_proj.calc_sampled_param_num()
        total_params += self.out_proj.calc_sampled_param_num()
        return total_params

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.k_proj.get_complexity(sequence_length)
        total_flops += self.v_proj.get_complexity(sequence_length)
        total_flops += self.q_proj.get_complexity(sequence_length)
        total_flops += self.out_proj.get_complexity(sequence_length)
        # attn
        swa = self.sample_attn_swz
        if swa == "global" or sequence_length <= swa // 2 + 1:
            swa = sequence_length
        total_flops += sequence_length * swa * self.sample_qkv_embed_dim
        # x
        total_flops += sequence_length * swa * self.sample_qkv_embed_dim
        return total_flops


if __name__ == "__main__":
    m = SMHA(768, 12)
    m.set_sample_config(768, 12, 768, 32)
    x = torch.empty((2, 5, 768))
    m(x, x, x)
