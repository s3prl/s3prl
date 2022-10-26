"""
Most of this code comes from the timm  library.
We tried to disentangle from the timm library version.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""
import collections
import logging
import math
import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers.vit_helpers import (
    DropPath,
    build_model_with_cfg,
    trunc_normal_,
    update_default_cfg_and_kwargs,
)

_logger = logging.getLogger()

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    "vit_tiny_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_tiny_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_small_patch32_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_small_patch32_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_small_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_small_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch32_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_base_patch32_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_base_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_large_patch32_224": _cfg(
        url="",  # no official model weights for this combo, only for in21k
    ),
    "vit_large_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_large_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_large_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    # patch models, imagenet21k (weights from official Google JAX impl)
    "vit_tiny_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_small_patch32_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_small_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_base_patch32_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_base_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_large_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth",
        num_classes=21843,
    ),
    "vit_large_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz",
        num_classes=21843,
    ),
    "vit_huge_patch14_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz",
        hf_hub="timm/vit_huge_patch14_224_in21k",
        num_classes=21843,
    ),
    # SAM trained models (https://arxiv.org/abs/2106.01548)
    "vit_base_patch32_sam_224": _cfg(
        url="https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz"
    ),
    "vit_base_patch16_sam_224": _cfg(
        url="https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz"
    ),
    # deit models (FB weights)
    "deit_tiny_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_small_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "deit_tiny_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    ),
    "deit_small_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    ),
    "deit_base_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    ),
    "deit_base_distilled_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384),
        crop_pct=1.0,
        classifier=("head", "head_dist"),
    ),
    # ViT ImageNet-21K-P pretraining by MILL
    "vit_base_patch16_224_miil_in21k": _cfg(
        url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth",
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_pct=0.875,
        interpolation="bilinear",
        num_classes=11221,
    ),
    "vit_base_patch16_224_miil": _cfg(
        url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm"
        "/vit_base_patch16_224_1k_miil_84_4.pth",
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_pct=0.875,
        interpolation="bilinear",
    ),
    # PaSST
    "passt_s_swa_p16_128_ap476": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/passt-s-f128-p16-s10-ap.476-swa.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 998),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=527,
    ),
    "passt_s_p16_s16_128_ap468": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s16-ap.468.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 998),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=527,
    ),
    "passt_s_swa_f128_stfthop100_p16_s10_ap473": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v0.0.3-audioset/passt-s-f128-stfthop100-p16-s10-ap.473-swa.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 3200),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=527,
    ),
    "passt_s_swa_f128_stfthop160_p16_s10_ap473": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v0.0.3-audioset/passt-s-f128-stfthop160-p16-s10-ap.473-swa.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 2000),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=527,
    ),
    "openmic2008_passt_u_f128_p16_s10_ap85_swa": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v0.0.4-openmic/openmic2008.passt-u-f128-p16-s10-ap.85-swa.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 3200),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=20,
    ),
    "openmic2008_passt_u_f128_p16_s10_ap85  ": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v0.0.4-openmic/openmic2008.passt-u-f128-p16-s10-ap.85.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 2000),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=20,
    ),
    "passt-s-f128-20sec-p16-s10-ap474-swa": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v0.0.5/passt-s-f128-20sec-p16-s10-ap.474-swa.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 2000),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=527,
    ),
    "passt-s-f128-30sec-p16-s10-ap473-swa": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v0.0.5/passt-s-f128-30sec-p16-s10-ap.473-swa.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 3000),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=527,
    ),
    "passt_b_f128_p16_s16_ap_459": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v.0.0.7-audioset/passt-b-f128-p16-s16-ap.459.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 998),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=527,
    ),
    "passt_u600_f128_p16_s16_ap_460": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v.0.0.7-audioset/passt-u600-f128-p16-s16-ap.460.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 998),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=527,
    ),
}


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = (
        conv_weight.float()
    )  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError("Weight format not supported by conversion.")
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= 3 / float(in_chans)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = (img_size[0] // stride[0], img_size[1] // stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # if not (H == self.img_size[0] and W == self.img_size[1]):
        #     warnings.warn(f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        # to do maybe replace weights
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PaSST(nn.Module):
    """

    Based on the implementation of Vision Transformer in timm library.
     Take a look at the get_model function, adapting the weights of pretrained imagenet models.

    """

    def __init__(
        self,
        u_patchout=0,
        s_patchout_t=0,
        s_patchout_f=0,
        img_size=(128, 998),
        patch_size=16,
        stride=16,
        in_chans=1,
        num_classes=527,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
    ):
        """
        Args:
            u_patchout: Unstructured Patchout integer, number of items to be removed from the final sequence
            s_patchout_t: structured Patchout time integer, number of columns to be removed from the patches grid
            s_patchout_f: structured Patchout Frequency integer, number of rows to be removed from the patches grid
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.u_patchout = u_patchout
        self.s_patchout_t = s_patchout_t
        self.s_patchout_f = s_patchout_f
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten=False,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        # PaSST
        # refer to https://arxiv.org/abs/2110.05069 Section 2
        self.new_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )  # for C and D tokens
        self.freq_new_pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, self.patch_embed.grid_size[0], 1)
        )  # | f
        self.time_new_pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, 1, self.patch_embed.grid_size[1])
        )  # __ t
        ####
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity(),
        )
        self.head_dist = None
        if distilled:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.new_pos_embed, std=0.02)
        trunc_normal_(self.freq_new_pos_embed, std=0.02)
        trunc_normal_(self.time_new_pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        if mode.startswith("jax"):
            # leave cls token as zeros to match jax impl
            raise RuntimeError("Not supported yet")
        else:
            trunc_normal_(self.cls_token, std=0.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "new_pos_embed",
            "freq_new_pos_embed",
            "time_new_pos_embed",
            "cls_token",
            "dist_token",
        }

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        if self.num_tokens == 2:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

    def forward_features(self, x):
        x = self.patch_embed(x)  # [b, e, f, t]
        B_dim, E_dim, F_dim, T_dim = x.shape  # slow
        # Adding Time/Freq information
        time_new_pos_embed = self.time_new_pos_embed
        if x.shape[-1] != time_new_pos_embed.shape[-1]:
            time_new_pos_embed = time_new_pos_embed[:, :, :, : x.shape[-1]]
        x = x + time_new_pos_embed
        x = x + self.freq_new_pos_embed

        # Structured Patchout https://arxiv.org/abs/2110.05069 Section 2.2
        if self.training and self.s_patchout_t:
            # ([1, 768, 1, 82])
            random_indices = (
                torch.randperm(T_dim)[: T_dim - self.s_patchout_t].sort().values
            )
            x = x[:, :, :, random_indices]
        if self.training and self.s_patchout_f:
            # [1, 768, 12, 1]
            random_indices = (
                torch.randperm(F_dim)[: F_dim - self.s_patchout_f].sort().values
            )
            x = x[:, :, random_indices, :]

        ###
        # Flatten the sequence
        x = x.flatten(2).transpose(1, 2)
        # Unstructured Patchout

        if self.training and self.u_patchout:
            seq_len = x.shape[1]
            random_indices = (
                torch.randperm(seq_len)[: seq_len - self.u_patchout].sort().values
            )
            x = x[:, random_indices, :]

        ####
        # Add the C/D tokens
        cls_tokens = self.cls_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, :1, :]
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            dist_token = (
                self.dist_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, 1:, :]
            )
            x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)

        if self.head_dist is not None:
            features = (x[0] + x[1]) / 2
            x = self.head(features)
            return x, features
        else:
            features = x
            x = self.head(x)

        return x, features


lecun_normal_ = None


def _init_vit_weights(
    module: nn.Module, name: str = "", head_bias: float = 0.0, jax_impl: bool = False
):
    """ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith("pre_logits"):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=(), mode="bicubic"):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info(
        "Resized position embedding: %s to %s with %s cls/dis tokens",
        posemb.shape,
        posemb_new.shape,
        num_tokens,
    )
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info("Position embedding grid-size from %s to %s", [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode=mode, align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def adapt_image_pos_embed_to_passt(posemb, num_tokens=1, gs_new=(), mode="bicubic"):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info(
        "Resized position embedding: %s to %s with %s cls/dis tokens",
        posemb.shape,
        gs_new,
        num_tokens,
    )
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))

    assert len(gs_new) >= 2
    _logger.info("Position embedding grid-size from %s to %s", [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode=mode, align_corners=False
    )
    freq_new_pos_embed = posemb_grid.mean(dim=3, keepdim=True)
    time_new_pos_embed = posemb_grid.mean(dim=2, keepdim=True)
    _logger.info("New Position cls/dstl embedding %s", posemb_tok.shape)
    _logger.info("New FREQ Position embedding %s", freq_new_pos_embed.shape)
    _logger.info("New TIME Position embedding %s", time_new_pos_embed.shape)
    return posemb_tok, freq_new_pos_embed, time_new_pos_embed


def checkpoint_filter_fn(state_dict, model):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    state_dict = {k: v for k, v in state_dict.items()}
    if "time_new_pos_embed" not in state_dict:
        # we are working with ImageNet model
        _logger.info("Adapting pos embedding from ImageNet pretrained model to PaSST.")
        v = state_dict.pop("pos_embed")
        (
            new_pos_embed,
            freq_new_pos_embed,
            time_new_pos_embed,
        ) = adapt_image_pos_embed_to_passt(
            v, getattr(model, "num_tokens", 1), model.patch_embed.grid_size
        )
        state_dict["new_pos_embed"] = new_pos_embed
        state_dict["freq_new_pos_embed"] = freq_new_pos_embed
        state_dict["time_new_pos_embed"] = time_new_pos_embed

    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # this should never occur
            v = resize_pos_embed(
                v,
                model.pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg["num_classes"]
    num_classes = kwargs.get("num_classes", default_num_classes)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        PaSST,
        variant,
        pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in default_cfg["url"],
        **kwargs,
    )
    return model


def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        representation_size=1280,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_huge_patch14_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "deit_base_distilled_patch16_384",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def passt_s_swa_p16_128_ap476(pretrained=False, **kwargs):
    """DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "passt_s_swa_p16_128_ap476",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def passt_s_p16_s16_128_ap468(pretrained=False, **kwargs):
    """PaSST pre-trained on AudioSet"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (16, 16):
        warnings.warn(
            f"This model was pre-trained with strides {(16, 16)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}."
        )
    model = _create_vision_transformer(
        "passt_s_p16_s16_128_ap468",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def passt_s_swa_f128_stfthop100_p16_s10_ap473(pretrained=False, **kwargs):
    """DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "passt_s_swa_f128_stfthop100_p16_s10_ap473",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def passt_s_swa_f128_stfthop160_p16_s10_ap473(pretrained=False, **kwargs):
    """DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "passt_s_swa_f128_stfthop160_p16_s10_ap473",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def openmic2008_passt_u_f128_p16_s10_ap85_swa(pretrained=False, **kwargs):
    """DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "openmic2008_passt_u_f128_p16_s10_ap85_swa",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def passt_s_f128_20sec_p16_s10_ap474_swa(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "passt-s-f128-20sec-p16-s10-ap474-swa",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def passt_s_f128_30sec_p16_s10_ap473_swa(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "passt-s-f128-30sec-p16-s10-ap473-swa",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def passt_b_f128_p16_s16_ap_459(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "passt_b_f128_p16_s16_ap_459",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def passt_u600_f128_p16_s16_ap_460(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "passt_u600_f128_p16_s16_ap_460",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


PatchEmbedAdaptiveMean = None
PatchEmbedAdaptiveMeanKeepConv = None


def fix_embedding_layer(model, embed="default"):
    if embed == "default":
        return model
    if embed == "overlap":
        model.patch_embed = PatchEmbedAdaptiveMean(replace=model.patch_embed)
    if embed == "am_keepconv":
        model.patch_embed = PatchEmbedAdaptiveMeanKeepConv(replace=model.patch_embed)
    return model


def get_model(
    arch="passt_s_swa_p16_128_ap476",
    pretrained=True,
    n_classes=527,
    in_channels=1,
    fstride=10,
    tstride=10,
    input_fdim=128,
    input_tdim=998,
    u_patchout=0,
    s_patchout_t=0,
    s_patchout_f=0,
):
    """
    :param arch: Base ViT or Deit architecture
    :param pretrained: use pretrained model on imagenet
    :param n_classes: number of classes
    :param in_channels: number of input channels: 1 for mono
    :param fstride: the patches stride over frequency.
    :param tstride: the patches stride over time.
    :param input_fdim: the expected input frequency bins.
    :param input_tdim: the expected input time bins.
    :param u_patchout: number of input patches to drop in Unstructured Patchout as defined in https://arxiv.org/abs/2110.05069
    :param s_patchout_t: number of input time frames to drop Structured Patchout as defined in https://arxiv.org/abs/2110.05069
    :param s_patchout_f:  number of input frequency bins to drop Structured Patchout as defined in https://arxiv.org/abs/2110.05069
    :param audioset_pretrain: use pretrained models on Audioset.
    :return:

    """
    model_func = None
    input_size = (input_fdim, input_tdim)
    stride = (fstride, tstride)
    if arch == "passt_deit_bd_p16_384":
        model_func = deit_base_distilled_patch16_384
    elif arch == "passt_s_swa_p16_128_ap476":  # pretrained
        model_func = passt_s_swa_p16_128_ap476
    elif arch == "passt_s_p16_s16_128_ap468":
        model_func = passt_s_p16_s16_128_ap468
    elif arch == "openmic2008":  # pretrained
        model_func = openmic2008_passt_u_f128_p16_s10_ap85_swa
    elif arch == "stfthop100":  # pretrained
        model_func = passt_s_swa_f128_stfthop100_p16_s10_ap473
    elif arch == "stfthop160":  # pretrained
        model_func = passt_s_swa_f128_stfthop160_p16_s10_ap473
    elif arch == "passt_20sec":  # pretrained
        model_func = passt_s_f128_20sec_p16_s10_ap474_swa
    elif arch == "passt_30sec":  # pretrained
        model_func = passt_s_f128_30sec_p16_s10_ap473_swa
    elif arch == "passt_u600_f128_p16_s16_ap_460":  # pretrained
        model_func = passt_u600_f128_p16_s16_ap_460
    elif arch == "passt_b_f128_p16_s16_ap_459":  # pretrained
        model_func = passt_b_f128_p16_s16_ap_459

    if model_func is None:
        raise RuntimeError(f"Unknown model {arch}")
    model = model_func(
        pretrained=pretrained,
        num_classes=n_classes,
        in_chans=in_channels,
        img_size=input_size,
        stride=stride,
        u_patchout=u_patchout,
        s_patchout_t=s_patchout_t,
        s_patchout_f=s_patchout_f,
    )
    model = fix_embedding_layer(model)
    return model
