# --------------------------------------------------------
# LightHuBERT: Lightweight and Configurable Speech Representation Learning with Once-for-All Hidden-Unit BERT (https://arxiv.org/pdf/2203.15610.pdf)
# Github source: https://github.com/mechanicalsea/lighthubert
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .modules.fairseq_modules import GradMultiply
from .modules.scaling_linear import SLinear
from .modules.scaling_multihead import SMHA
from .modules.scaling_transformer import STransformerEncoder
from .modules.w2v2_modules import ConvFeatureExtractionModel

logger = logging.getLogger(__name__)


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )

        mask[i, mask_idc] = True

    return mask


class LightHuBERTSupernetConfig(object):
    """LightHuBERT search space providing supernet, search space, a subnet."""

    def __init__(self, supernet_type="base"):
        assert supernet_type.lower() in ["base", "small"]
        self.supernet_type = supernet_type.lower()

        def prod(x: List):
            ans = 1
            for xi in x:
                ans *= len(set(xi))
            return ans

        self.search_space_size = self.search_space["embed_dim"].__len__() * (
            sum(
                len(self.search_space["ffn_ratio"]) ** di
                * len(self.search_space["heads_num"]) ** di
                * prod(
                    self.search_space["slide_wsz"][:di]
                    if "slide_wsz" in self.search_space
                    else [["global"] for _ in range(di)]
                )
                for di in self.search_space["layer_num"]
            )
        )

    @property
    def supernet(self):
        return {
            "atten_dim": 768,
            "embed_dim": 768,
            "ffn_ratio": 4.0,
            "heads_num": 12,
            "layer_num": 12,
        }

    @property
    def search_space(self):
        if self.supernet_type == "base":
            return {
                "atten_dim": [512, 640, 768],
                "embed_dim": [512, 640, 768],
                "ffn_ratio": [3.5, 4.0],
                "heads_num": [8, 10, 12],
                "layer_num": [12],
            }
        elif self.supernet_type == "small":
            return {
                "atten_dim": [256, 384, 512],
                "embed_dim": [256, 384, 512],
                "ffn_ratio": [3.0, 3.5, 4.0],
                "heads_num": [4, 6, 8],
                "layer_num": [10, 11, 12],
            }

    @property
    def subnet(self):
        if self.supernet_type == "base":
            return {
                "atten_dim": [640 for _ in range(12)],
                "embed_dim": 640,
                "ffn_ratio": [4.0 for _ in range(12)],
                "ffn_embed": [2560 for _ in range(12)],
                "heads_num": [10 for _ in range(12)],
                "layer_num": 12,
                "slide_wsz": ["global" for _ in range(12)],
            }
        elif self.supernet_type == "small":
            return {
                "atten_dim": [384 for _ in range(12)],
                "embed_dim": 384,
                "ffn_ratio": [4.0 for _ in range(12)],
                "ffn_embed": [1536 for _ in range(12)],
                "heads_num": [6 for _ in range(12)],
                "layer_num": 12,
                "slide_wsz": ["global" for _ in range(12)],
            }

    @property
    def dimensions(self) -> list:
        return ["atten_dim", "embed_dim", "ffn_ratio", "heads_num", "layer_num"]

    @property
    def num_subnets(self) -> int:
        """Count the number of subnets in the supernet."""
        return self.search_space_size

    @property
    def slide_mode(self) -> str:
        return "stride"

    @property
    def max_subnet(self) -> dict:
        layer_num = max(self.search_space["layer_num"])
        embed_dim = max(self.search_space["embed_dim"])
        ffn_embed = [
            int(max(self.search_space["ffn_ratio"]) * embed_dim)
            for _ in range(layer_num)
        ]
        heads_num = [max(self.search_space["heads_num"]) for _ in range(layer_num)]
        atten_dim = [heads_num[i] * 64 for i in range(layer_num)]
        slide_wsz = ["global" for _ in range(layer_num)]
        return {
            "atten_dim": atten_dim,  # List[int]
            "embed_dim": embed_dim,  # int
            "ffn_embed": ffn_embed,  # List[int]
            "heads_num": heads_num,  # List[int]
            "layer_num": layer_num,  # int
            "slide_wsz": slide_wsz,  # List[int] or List[str]
        }

    @property
    def min_subnet(self) -> dict:
        assert self.search_space is not None
        assert self.search_space is not None
        layer_num = min(self.search_space["layer_num"])
        embed_dim = min(self.search_space["embed_dim"])
        ffn_embed = [
            int(min(self.search_space["ffn_ratio"]) * embed_dim)
            for _ in range(layer_num)
        ]
        heads_num = [min(self.search_space["heads_num"]) for _ in range(layer_num)]
        atten_dim = [heads_num[i] * 64 for i in range(layer_num)]
        slide_wsz = ["global" for _ in range(layer_num)]
        return {
            "atten_dim": atten_dim,  # List[int]
            "embed_dim": embed_dim,  # int
            "ffn_embed": ffn_embed,  # List[int]
            "heads_num": heads_num,  # List[int]
            "layer_num": layer_num,  # int
            "slide_wsz": slide_wsz,  # List[int] or List[str]
        }

    def sample_subnet(self) -> dict:
        """sample a subnet for search space, e.g.,
        {
            "atten_dim": [512, 512, 512, 512, 512, 512]        # List[int]
            "embed_dim": 512,                                  # int
            # "ffn_ratio": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]      # List[int]
            "ffn_embed": [1536, 1536, 1536, 1536, 1536, 1536]  # List[int]
            "heads_num": [8, 8, 8, 8, 8, 8, 8],                # List[int]
            "layer_num": 6,                                    # int
            "slide_wsz": [global, global, global, global, global, global], # List[int] or List[str]
        }
        ffn_ratio 3.0 -> ffn_embed 1536
        heads_num 8 -> atten_dim 512
        """
        assert self.search_space is not None
        layer_num = random.choice(self.search_space["layer_num"])
        embed_dim = random.choice(self.search_space["embed_dim"])
        ffn_embed = [
            int(random.choice(self.search_space["ffn_ratio"]) * embed_dim)
            for _ in range(layer_num)
        ]
        heads_num = [
            random.choice(self.search_space["heads_num"]) for _ in range(layer_num)
        ]
        atten_dim = [heads_num[i] * 64 for i in range(layer_num)]
        if "slide_wsz" in self.search_space:
            slide_wsz = [
                random.choice(self.search_space["slide_wsz"][i])
                for i in range(layer_num)
            ]
            slide_wsz = [
                int(swsz) if swsz != "global" else str(swsz) for swsz in slide_wsz
            ]
        else:
            slide_wsz = ["global" for _ in range(layer_num)]
        return {
            "atten_dim": atten_dim,  # List[int]
            "embed_dim": embed_dim,  # int
            "ffn_embed": ffn_embed,  # List[int]
            "heads_num": heads_num,  # List[int]
            "layer_num": layer_num,  # int
            "slide_wsz": slide_wsz,  # List[int] or List[str]
        }


class LightHuBERTConfig:
    def __init__(self, cfg=None):
        self.supernet_type: str = "base"  # 'base' or 'small' supernet

        self.prune_encoder_pos_conv: bool = True  # if true, prune multi-layer position convolution totally, otherwise, prune last layer to encoder

        # Configuration for Student
        self.teacher_embed_dim: int = 768  # embedding size of teacher's outputs
        self.layer_pred_num: str = "0,0,0,0,0,0,0,0,0,0,0,1"  # specify the predicted number of each layer, e.g., '2,2,2,2,2,2'

        # add pos_conv_depth to hubert as wav2vec2 and enable data2vec architecture
        self.pos_conv_depth: int = 1  # depth of positional encoder network

        # hubert config
        self.label_rate: int = 50
        self.extractor_mode: str = "layer_norm"  # mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with normalize=True)
        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use
        self.layer_type: str = "transformer"  # layer type in encoder

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = (
            0.0  # dropout probability after activation in FFN
        )
        self.encoder_layerdrop: float = (
            0.0  # probability of dropping a tarnsformer layer
        )
        self.dropout_input: float = (
            0.0  # dropout to apply to the input (after feat extr)
        )
        self.dropout_features: float = (
            0.0  # dropout to apply to the features (after feat extr)
        )

        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"  # string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]
        self.conv_bias: bool = False  # include bias in conv encoder
        self.feature_grad_mult: float = (
            1.0  # multiply feature extractor var grads by this
        )

        # masking
        self.mask_length: int = 10  # mask length"
        self.mask_prob: float = 0.65  # probability of replacing a token with mask
        self.mask_selection: str = "static"  # how to choose mask length
        self.mask_other: float = 0  # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_overlap: bool = False  # whether to allow masks to overlap
        self.mask_min_space: int = (
            1  # min space between spans (if no overlap is enabled)
        )

        # channel masking
        self.mask_channel_length: int = 10  # length of the mask for features (channels)
        self.mask_channel_prob: float = 0.0  # probability of replacing a feature with 0
        self.mask_channel_selection: str = (
            "static"  # how to choose mask length for channel masking
        )
        self.mask_channel_other: float = 0  # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_channel_overlap: bool = (
            False  # whether to allow channel masks to overlap
        )
        self.mask_channel_min_space: int = (
            1  # min space between spans (if no overlap is enabled)
        )

        # positional embeddings
        self.conv_pos: int = (
            128  # number of filters for convolutional positional embeddings
        )
        self.conv_pos_groups: int = (
            16  # number of groups for convolutional positional embedding
        )

        # FP16 optimization
        self.required_seq_len_multiple: int = 2  # pad the input to encoder such that the sequence length is divisible by multiple

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        for ki, vi in cfg.items():
            if ki in self.__dict__:
                self.__dict__.update({ki: vi})


class LightHuBERT(nn.Module):
    def __init__(self, cfg: LightHuBERTConfig):
        super().__init__()
        logger.info(f"LightHuBERT Config: {cfg}")
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            SLinear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )  # lighthubert component

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
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

        self.encoder = STransformerEncoder(cfg)  # lighthubert component
        self.layer_norm = nn.LayerNorm(self.embed)

        # student config
        self.encoder_layers = cfg.encoder_layers
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.teacher_embed_dim = cfg.teacher_embed_dim

        layer_pred_num = [max(int(n), 0) for n in cfg.layer_pred_num.split(",")]
        assert len(layer_pred_num) == self.encoder_layers, f"{len(layer_pred_num)}"
        self.layer_pred_heads = torch.nn.ModuleDict()
        for layer_i in range(self.encoder_layers):
            if layer_pred_num[layer_i] > 0:
                self.layer_pred_heads[f"{layer_i}"] = SLinear(
                    self.encoder_embed_dim,
                    self.teacher_embed_dim * layer_pred_num[layer_i],
                    bias=False,
                )
        self.layer_pred_num = layer_pred_num
        logger.info(f"predicting heads: {layer_pred_num}")

        # lighthubert config
        assert cfg.encoder_attention_heads * 64 == cfg.encoder_embed_dim
        self.sample_layer_num = cfg.encoder_layers
        self.sample_embed_dim = cfg.encoder_embed_dim
        self.sample_ffn_embed = cfg.encoder_ffn_embed_dim
        self.sample_heads_num = cfg.encoder_attention_heads
        self.sample_atten_dim = cfg.encoder_embed_dim
        self.sample_slide_wsz = None

        def subnet_params(subnet):
            self.set_sample_config(subnet)
            return self.calc_sampled_param_num()

        self.supernet = LightHuBERTSupernetConfig(cfg.supernet_type)
        self._switch_slide_attention()
        self.dynamize(False)

        # search space
        logger.info(
            f"search space ({self.supernet.num_subnets:,} subnets): {self.supernet.search_space}"
        )
        # min subnet
        total_params = subnet_params(self.supernet.min_subnet)
        logger.info(
            f"min subnet ({total_params/1e6:.0f} Params): {self.supernet.min_subnet}"
        )
        # max subnet
        total_params = subnet_params(self.supernet.max_subnet)
        logger.info(
            f"max subnet ({total_params/1e6:.0f} Params): {self.supernet.max_subnet}"
        )

    def _switch_slide_attention(self, slide_mode="stride"):
        """Set sliding attention manner to either stride or mask."""
        assert slide_mode in ["stride", "mask"]
        for n, m in self.named_modules():
            if isinstance(m, SMHA):
                assert hasattr(m, "set_slide_mode")
                m.set_slide_mode(slide_mode)

    def dynamize(self, mode: bool = True, log_subnet: bool = True):
        """To determine whether to sample a subnet during forward.
        If not training, self.dynamize(False)
        Else: self.dynamize()

        self.feature_extractor: ConvFeatureExtractionModel
        self.post_extract_proj: nn.Linear
        self.mask_emb: nn.Parameter
        self.encoder: TransformerEncoder
        self.layer_norm: nn.LayerNorm
        """
        self.dynamic = mode
        self.verbose = log_subnet & (self.supernet.num_subnets > 1)
        if not self.dynamic:
            subnet = self.supernet.subnet
            self.set_sample_config(subnet)
            if hasattr(self, "handle"):
                self.handle.remove()
            return

        # sample a subnet before forward
        def set_subnet(module, input):
            if not getattr(module, "dynamic", False):
                return
            assert hasattr(module, "set_sample_config")
            subnet = module.supernet.sample_subnet()
            module.set_sample_config(subnet)
            if getattr(module, "verbose", False):
                total_params = module.calc_sampled_param_num() / 1e6
                logger.info(f"dynamic subnet ({total_params:.2f}M Params): {subnet}")

        self.handle = self.register_forward_pre_hook(set_subnet)

    def set_sample_config(self, config: dict):
        """
        config: {
            "atten_dim": atten_dim, # List[int]
            "embed_dim": embed_dim, # int
            "ffn_embed": ffn_embed, # List[int]
            "heads_num": heads_num, # List[int]
            "layer_num": layer_num, # int
            "slide_wsz": slide_wsz, # List[int] or List[str]
        }
        """
        self.sample_layer_num = config["layer_num"]
        self.sample_atten_dim = config["atten_dim"]
        self.sample_embed_dim = config["embed_dim"]
        self.sample_ffn_embed = config["ffn_embed"]
        self.sample_heads_num = config["heads_num"]
        self.sample_slide_wsz = config["slide_wsz"]
        self._sample_parameters()

    def _sample_parameters(self):
        self.post_extract_proj.set_sample_config(
            self.embed,
            self.sample_embed_dim,
        )
        self.encoder.set_sample_config(
            self.sample_layer_num,
            self.sample_atten_dim,
            self.sample_embed_dim,
            self.sample_ffn_embed,
            self.sample_heads_num,
            self.sample_slide_wsz,
        )
        if self.layer_pred_heads is not None:
            for _, lrhi in self.layer_pred_heads.items():
                lrhi.set_sample_config(self.sample_embed_dim, self.teacher_embed_dim)

    def calc_sampled_param_num(self):
        total_params = 0
        for n, p in self.named_parameters():
            if (
                not n.startswith("post_extract_proj")
                and not n.startswith("encoder")
                and not n.startswith("layer_pred_heads")
            ):
                total_params += p.numel()
        total_params += self.post_extract_proj.calc_sampled_param_num()
        total_params += self.encoder.calc_sampled_param_num()
        # not consider prediction head
        # if self.layer_pred_heads is not None:
        #     for _, lrhi in self.layer_pred_heads.items():
        #         total_params += lrhi.calc_sampled_param_num()
        return total_params

    def apply_mask(self, x, padding_mask, target_list, mask_indices=None):
        """Refactor mask method to enable to mask with the given masking indices."""
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
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb[:C]
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
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

        return x, mask_indices

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        mask_indices=None,
    ) -> Dict[str, torch.Tensor]:
        """Refactor extract feature method to enable to output
        1. hidden representations
        2. mask indices corresponding to masked language modeling
        """
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask, target_list, mask_indices=mask_indices
            )
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None,
        )

        layer_head_list = [
            None,
        ] * self.encoder_layers
        if self.layer_pred_heads is not None:
            if (
                len(self.layer_pred_heads) == 1
                and f"{self.encoder_layers - 1}" in self.layer_pred_heads
            ):
                # enable layerdrop in training while only consider the last hidden state
                _layer_results = [
                    (None, None, None) for _ in range(self.encoder_layers - 1)
                ] + [(x.transpose(0, 1), None, None)]
            else:
                if len(layer_results) == self.encoder_layers:
                    _layer_results = layer_results
                else:
                    raise ValueError("Multi-layer prediction requires 0.0 layerdrop.")
            for layer_i, (layer_hs, _, _) in enumerate(_layer_results):
                if f"{layer_i}" in self.layer_pred_heads:
                    layer_pred_head = self.layer_pred_heads[f"{layer_i}"]
                    # enable hidden states with variable embedding size
                    layer_pred_head.set_sample_config(
                        layer_hs.size(-1), layer_pred_head.out_features
                    )
                    layer_output = (
                        layer_pred_head(layer_hs).transpose(0, 1).contiguous()
                    )
                    layer_head_list[layer_i] = torch.split(
                        layer_output, self.teacher_embed_dim, dim=-1
                    )

        hidden_states = [features] + [
            hs.transpose(0, 1).contiguous() for hs, attn, ls in layer_results
        ]
        attn_matrices = [
            attn for hs, attn, ls in layer_results
        ]  # List[(heads_num, batch_size, tgt_len, src_len)]

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "hidden_states": hidden_states,
                "layer_heads": layer_head_list,
            }

        result = {
            "padding_mask": padding_mask,
            "features_pen": features_pen,
            "x": x,
            "hidden_states": hidden_states,
            "attn_matrices": attn_matrices,
            "layer_heads": layer_head_list,
            "mask_indices": mask_indices,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        ret_hs: bool = False,
        cat_heads: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refactor extract feature method to enable to output hidden representations."""
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        if ret_hs:
            feature = res["hidden_states"]
            if cat_heads:
                layer_heads_list = [
                    layer_heads
                    for layer_heads in res["layer_heads"]
                    if layer_heads is not None
                ]
                for layer_heads in layer_heads_list:
                    feature += layer_heads
        elif ret_conv:
            feature = res["hidden_states"][0]
        else:
            feature = res["x"]
        return feature, res["padding_mask"]

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.extractor_frozen = True

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            if self.feature_extractor.parameters().__next__().requires_grad:
                extra_losses.append(net_output["features_pen"])
                names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        """Meantime remove predicting heads"""
        self.layer_pred_heads = None
        self.layer_pred_num = [
            0,
        ] * self.encoder_layers
