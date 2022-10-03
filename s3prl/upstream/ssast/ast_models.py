# -*- coding: utf-8 -*-
# @Time    : 7/16/21 3:12 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

# the unified ast models for all pretraining/fine-tuning tasks.

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ASTModel(nn.Module):
    def __init__(
        self,
        label_dim=527,
        fshape=128,
        tshape=2,
        fstride=128,
        tstride=2,
        input_fdim=128,
        input_tdim=1024,
        model_size="base",
        pretrain_stage=True,
        load_pretrained_mdl_path=None,
    ):

        super(ASTModel, self).__init__()
        assert (
            timm.__version__ == "0.4.5"
        ), "Please use timm == 0.4.5, the code might not be compatible with newer versions."

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # pretrain the AST models
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError(
                    "Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None."
                )
            if fstride != fshape or tstride != tshape:
                raise ValueError(
                    "fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported."
                )

            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == "tiny":
                self.v = timm.create_model(
                    "vit_deit_tiny_distilled_patch16_224", pretrained=False
                )
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == "small":
                self.v = timm.create_model(
                    "vit_deit_small_distilled_patch16_224", pretrained=False
                )
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == "base":
                self.v = timm.create_model(
                    "vit_deit_base_distilled_patch16_384", pretrained=False
                )
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == "base_nokd":
                self.v = timm.create_model(
                    "vit_deit_base_patch16_384", pretrained=False
                )
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception(
                    "Model size must be one of tiny, small, base, base_nokd"
                )

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches**0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # SSL Pretraining Code
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(
                torch.tensor(input_fdim), requires_grad=False
            ), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            # masked patch classification (discriminative objective) layer
            # we use two layers for pretext task, but using a single layer has similar performance.
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            self.cpredlayer = nn.Sequential(
                nn.Linear(self.original_embedding_dim, self.original_embedding_dim),
                nn.ReLU(),
                nn.Linear(self.original_embedding_dim, 256),
            )
            # masked patch reconstruction (generative objective) layer
            self.gpredlayer = nn.Sequential(
                nn.Linear(self.original_embedding_dim, self.original_embedding_dim),
                nn.ReLU(),
                nn.Linear(self.original_embedding_dim, 256),
            )
            self.unfold = torch.nn.Unfold(
                kernel_size=(fshape, tshape), stride=(fstride, tstride)
            )

            # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
            self.mask_embed = nn.Parameter(
                torch.zeros([1, 1, self.original_embedding_dim])
            )
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # get the intermediate shape
            self.p_f_dim, self.p_t_dim = self.get_shape(
                fstride, tstride, input_fdim, input_tdim, fshape, tshape
            )
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print(
                "pretraining patch split stride: frequency={:d}, time={:d}".format(
                    fstride, tstride
                )
            )
            print(
                "pretraining patch shape: frequency={:d}, time={:d}".format(
                    fshape, tshape
                )
            )
            print(
                "pretraining patch array dimension: frequency={:d}, time={:d}".format(
                    self.p_f_dim, self.p_t_dim
                )
            )
            print("pretraining number of patches={:d}".format(num_patches))

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(
                1,
                self.original_embedding_dim,
                kernel_size=(fshape, tshape),
                stride=(fstride, tstride),
            )
            self.v.patch_embed.proj = new_proj

            # use trainable positional embedding
            new_pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    self.v.patch_embed.num_patches + self.cls_token_num,
                    self.original_embedding_dim,
                )
            )
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=0.02)

        # use a pretrained models for finetuning
        elif pretrain_stage == False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if load_pretrained_mdl_path == None:
                raise ValueError(
                    "Please set load_pretrained_mdl_path to load a pretrained models."
                )
            sd = torch.load(load_pretrained_mdl_path, map_location=device)
            # get the fshape and tshape, input_fdim and input_tdim in the pretraining stage
            try:
                p_fshape, p_tshape = (
                    sd["module.v.patch_embed.proj.weight"].shape[2],
                    sd["module.v.patch_embed.proj.weight"].shape[3],
                )
                p_input_fdim, p_input_tdim = (
                    sd["module.p_input_fdim"].item(),
                    sd["module.p_input_tdim"].item(),
                )
            except:
                raise ValueError(
                    "The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again."
                )

            print("now load a SSL pretrained models from " + load_pretrained_mdl_path)
            # during pretraining, fstride=fshape and tstride=tshape because no patch overlapping is used
            # here, input_fdim and input_tdim should be that used in pretraining, not that in the fine-tuning.
            # we need to know input_fdim and input_tdim to do positional embedding cut/interpolation.
            # generally it should be better to use same input_fdim during pretraining and finetuning, but input_tdim can be safely different
            audio_model = ASTModel(
                fstride=p_fshape,
                tstride=p_tshape,
                fshape=p_fshape,
                tshape=p_tshape,
                input_fdim=p_input_fdim,
                input_tdim=p_input_tdim,
                pretrain_stage=True,
                model_size=model_size,
            )
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            # mlp head for fine-tuning
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            f_dim, t_dim = self.get_shape(
                fstride, tstride, input_fdim, input_tdim, fshape, tshape
            )
            # patch array dimension during pretraining
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches
            print(
                "fine-tuning patch split stride: frequncey={:d}, time={:d}".format(
                    fstride, tstride
                )
            )
            print("fine-tuning number of patches={:d}".format(num_patches))

            # patch shape should be same for pretraining and fine-tuning
            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError(
                    "The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}".format(
                        p_fshape, p_tshape, fshape, tshape
                    )
                )

            # patch split stride generally should be different for pretraining and fine-tuning, as patch split overlapping is only used in finetuning
            # during pretraining, p_fshape = p_fstride and p_tshape = p_tstride
            if fstride != p_fshape or tstride != p_tshape:
                # initialize a new patch embedding layer with desired new stride.
                new_proj = torch.nn.Conv2d(
                    1,
                    self.original_embedding_dim,
                    kernel_size=(fshape, tshape),
                    stride=(fstride, tstride),
                )
                # but the weights of patch embedding layer is still got from the pretrained models
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj

            new_pos_embed = (
                self.v.pos_embed[:, self.cls_token_num :, :]
                .detach()
                .reshape(1, p_num_patches, self.original_embedding_dim)
                .transpose(1, 2)
                .reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            )
            # cut or interpolate the positional embedding
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[
                    :,
                    :,
                    :,
                    int(p_t_dim / 2)
                    - int(t_dim / 2) : int(p_t_dim / 2)
                    - int(t_dim / 2)
                    + t_dim,
                ]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(8, t_dim), mode="bilinear"
                )
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[
                    :,
                    :,
                    int(p_f_dim / 2)
                    - int(f_dim / 2) : int(p_f_dim / 2)
                    - int(f_dim / 2)
                    + t_dim,
                    :,
                ]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                )

            new_pos_embed = new_pos_embed.reshape(
                1, self.original_embedding_dim, num_patches
            ).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat(
                    [
                        self.v.pos_embed[:, : self.cls_token_num, :].detach(),
                        new_pos_embed,
                    ],
                    dim=1,
                )
            )

    # get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(fshape, tshape),
            stride=(fstride, tstride),
        )
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        # superb needs represents of each layer
        reps = []
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
            reps.append(x)
        x = self.v.norm(x)

        # reps=representation of each layer, x=final representation
        return reps, x
