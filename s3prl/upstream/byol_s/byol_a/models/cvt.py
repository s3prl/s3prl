"""
Convolutional Transformer

Largely inspired by https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cvt.py

Original paper:
@article{wu2021cvt,
  title={Cvt: Introducing convolutions to vision transformers},
  author={Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei},
  journal={arXiv preprint arXiv:2103.15808},
  year={2021}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix) :], x[1]), tuple(kwargs_with_prefix.items()))
    )
    return kwargs_without_prefix, kwargs


# classes


class LayerNorm(nn.Module):
    """Layer normalization, but done in channel dimension #1"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    """Pre-Normalization layer"""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    """Convolutional projection in the transformer."""

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),  # 1x1 convolution
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),  # 1x1 convolution
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DepthWiseConv2d(nn.Module):
    """Depthwise convolutional layer"""

    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Custom Attention layer"""

    def __init__(
        self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_q = DepthWiseConv2d(
            dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False
        )
        self.to_kv = DepthWiseConv2d(
            dim,
            inner_dim * 2,
            proj_kernel,
            padding=padding,
            stride=kv_proj_stride,
            bias=False,
        )

        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout))

    def forward(self, x):
        from einops import rearrange, repeat
        from einops.layers.torch import Rearrange

        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> (b h) (x y) d", h=h), (q, k, v)
        )

        dots = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = self.attend(dots)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) (x y) d -> b (h d) x y", h=h, y=y)
        return self.to_out(out)


class Transformer(nn.Module):
    """Custom Transformer layer."""

    def __init__(
        self,
        dim,
        proj_kernel,
        kv_proj_stride,
        depth,
        heads,
        dim_head=64,
        mlp_mult=4,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                proj_kernel=proj_kernel,
                                kv_proj_stride=kv_proj_stride,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CvT(nn.Module):
    """Convolutional Transformer module.

    Adapted for self-supervised training

    Attributes
    ----------
    s{i}_emb_dim: int
        Embedding dimention at stage i

    s{i}_emb_kernel: int
        Convolutional kernel size at stage i

    s{i}_emb_stride: int
        Convolutional stride at stage i

    s{i}_kv_proj_stride: int
        Convolutional stride in the convolutional projection layers at stage i

    s{i}_heads: int
        Number of attention heads at stage i

    s{i}_depth: int
        Transformer depth at stage i

    s{i}_mlp_mult: int
        MLP ratio at stage i

    dropout: float
        Dropout ratio
    """

    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = 16000
    embedding_size = 2048
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    def __init__(
        self,
        *,
        s1_emb_dim=64,
        s1_emb_kernel=7,
        s1_emb_stride=4,
        s1_proj_kernel=3,
        s1_kv_proj_stride=2,
        s1_heads=1,
        s1_depth=1,
        s1_mlp_mult=4,
        s2_emb_dim=192,
        s2_emb_kernel=3,
        s2_emb_stride=2,
        s2_proj_kernel=3,
        s2_kv_proj_stride=2,
        s2_heads=3,
        s2_depth=2,
        s2_mlp_mult=4,
        s3_emb_dim=384,
        s3_emb_kernel=3,
        s3_emb_stride=2,
        s3_proj_kernel=3,
        s3_kv_proj_stride=2,
        s3_heads=6,
        s3_depth=10,
        s3_mlp_mult=4,
        dropout=0.0,
        pool="mean",
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = 1
        layers = []

        for prefix in ("s1", "s2", "s3"):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f"{prefix}_", kwargs)

            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        config["emb_dim"],
                        kernel_size=config["emb_kernel"],
                        padding=(config["emb_kernel"] // 2),
                        stride=config["emb_stride"],
                    ),
                    LayerNorm(config["emb_dim"]),
                    Transformer(
                        dim=config["emb_dim"],
                        proj_kernel=config["proj_kernel"],
                        kv_proj_stride=config["kv_proj_stride"],
                        depth=config["depth"],
                        heads=config["heads"],
                        mlp_mult=config["mlp_mult"],
                        dropout=dropout,
                    ),
                )
            )

            dim = config["emb_dim"]

        self.pool = pool
        assert self.pool in ["mean", "max", "mean+max"]

        if self.pool == "mean":
            self.pool_layers = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        elif self.pool == "max":
            self.pool_layers = (nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten()),)
        else:
            self.pool_layers = nn.Sequential(nn.Identity())

        self.layers = nn.Sequential(
            *layers,
            *self.pool_layers,
        )

    def forward(self, x):
        x = self.layers(x)

        if self.pool == "mean+max":
            x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
            B, T, D, C = x.shape
            x = x.reshape((B, T, C * D))  # (batch, time, mel*ch)
            x = x.mean(1) + x.amax(1)
        return x
