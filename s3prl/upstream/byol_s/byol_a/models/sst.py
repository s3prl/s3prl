"""Largely inspired by https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py"""
import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange


class SST(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        channels=1,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, embed_dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, embed_dim)
        )  # Learnable embedding
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * num_heads,
            dropout=dropout,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # No need for cls_token? We're not doing classification
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, : (n + 1)]

        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(1)  # https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

        x = x.mean(1) + x.amax(1)  # BYOL-A mean + max

        return x
