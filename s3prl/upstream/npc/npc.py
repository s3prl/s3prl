# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/npc/npc.py ]
#   Synopsis     [ the npc model]
#   Author       [ Alexander H. Liu (https://github.com/Alexander-H-Liu) ]
#   Reference    [ https://github.com/Alexander-H-Liu/NPC ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import copy

import torch
import torch.nn as nn

from .vq import VQLayer


class MaskConvBlock(nn.Module):
    """Masked Convolution Blocks as described in NPC paper"""

    def __init__(self, input_size, hidden_size, kernel_size, mask_size):
        super(MaskConvBlock, self).__init__()
        assert kernel_size - mask_size > 0, "Mask > kernel somewhere in the model"
        # CNN for computing feature (ToDo: other activation?)
        self.act = nn.Tanh()
        self.pad_size = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=self.pad_size,
        )
        # Fixed mask for NPC
        mask_head = (kernel_size - mask_size) // 2
        mask_tail = mask_head + mask_size
        conv_mask = torch.ones_like(self.conv.weight)
        conv_mask[:, :, mask_head:mask_tail] = 0
        self.register_buffer("conv_mask", conv_mask)

    def forward(self, feat):
        feat = nn.functional.conv1d(
            feat,
            self.conv_mask * self.conv.weight,
            bias=self.conv.bias,
            padding=self.pad_size,
        )
        feat = feat.permute(0, 2, 1)  # BxCxT -> BxTxC
        feat = self.act(feat)
        return feat


class ConvBlock(nn.Module):
    """Convolution Blocks as described in NPC paper"""

    def __init__(
        self, input_size, hidden_size, residual, dropout, batch_norm, activate
    ):
        super(ConvBlock, self).__init__()
        self.residual = residual
        if activate == "relu":
            self.act = nn.ReLU()
        elif activate == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError
        self.conv = nn.Conv1d(
            input_size, hidden_size, kernel_size=3, stride=1, padding=1
        )
        self.linear = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat):
        res = feat
        out = self.conv(feat)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.act(out)
        out = self.linear(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.dropout(out)
        if self.residual:
            out = out + res
        return self.act(out)


class NPC(nn.Module):
    """NPC model with stacked ConvBlocks & Masked ConvBlocks"""

    def __init__(
        self,
        input_size,
        hidden_size,
        n_blocks,
        dropout,
        residual,
        kernel_size,
        mask_size,
        vq=None,
        batch_norm=True,
        activate="relu",
        disable_cross_layer=False,
        dim_bottleneck=None,
    ):
        super(NPC, self).__init__()

        # Setup
        assert kernel_size % 2 == 1, "Kernel size can only be odd numbers"
        assert mask_size % 2 == 1, "Mask size can only be odd numbers"
        assert n_blocks >= 1, "At least 1 block needed"
        self.code_dim = hidden_size
        self.n_blocks = n_blocks
        self.input_mask_size = mask_size
        self.kernel_size = kernel_size
        self.disable_cross_layer = disable_cross_layer
        self.apply_vq = vq is not None
        self.apply_ae = dim_bottleneck is not None
        if self.apply_ae:
            assert not self.apply_vq
            self.dim_bottleneck = dim_bottleneck

        # Build blocks
        self.blocks, self.masked_convs = [], []
        cur_mask_size = mask_size
        for i in range(n_blocks):
            h_dim = input_size if i == 0 else hidden_size
            res = False if i == 0 else residual
            # ConvBlock
            self.blocks.append(
                ConvBlock(h_dim, hidden_size, res, dropout, batch_norm, activate)
            )
            # Masked ConvBlock on each or last layer
            cur_mask_size = cur_mask_size + 2
            if self.disable_cross_layer and (i != (n_blocks - 1)):
                self.masked_convs.append(None)
            else:
                self.masked_convs.append(
                    MaskConvBlock(hidden_size, hidden_size, kernel_size, cur_mask_size)
                )
        self.blocks = nn.ModuleList(self.blocks)
        self.masked_convs = nn.ModuleList(self.masked_convs)

        # Creates N-group VQ
        if self.apply_vq:
            self.vq_layers = []
            vq_config = copy.deepcopy(vq)
            codebook_size = vq_config.pop("codebook_size")
            self.vq_code_dims = vq_config.pop("code_dim")
            assert len(self.vq_code_dims) == len(codebook_size)
            assert sum(self.vq_code_dims) == hidden_size
            for cs, cd in zip(codebook_size, self.vq_code_dims):
                self.vq_layers.append(
                    VQLayer(input_size=cd, code_dim=cd, codebook_size=cs, **vq_config)
                )
            self.vq_layers = nn.ModuleList(self.vq_layers)

        # Back to spectrogram
        if self.apply_ae:
            self.ae_bottleneck = nn.Linear(hidden_size, self.dim_bottleneck, bias=False)
            self.postnet = nn.Linear(self.dim_bottleneck, input_size)
        else:
            self.postnet = nn.Linear(hidden_size, input_size)

    def create_msg(self):
        msg_list = []
        msg_list.append(
            "Model spec.| Method = NPC\t| # of Blocks = {}\t".format(self.n_blocks)
        )
        msg_list.append(
            "           | Desired input mask size = {}".format(self.input_mask_size)
        )
        msg_list.append(
            "           | Receptive field size = {}".format(
                self.kernel_size + 2 * self.n_blocks
            )
        )
        return msg_list

    def report_ppx(self):
        """Returns perplexity of VQ distribution"""
        if self.apply_vq:
            # ToDo: support more than 2 groups
            rt = [vq_layer.report_ppx() for vq_layer in self.vq_layers] + [None]
            return rt[0], rt[1]
        else:
            return None, None

    def report_usg(self):
        """Returns usage of VQ codebook"""
        if self.apply_vq:
            # ToDo: support more than 2 groups
            rt = [vq_layer.report_usg() for vq_layer in self.vq_layers] + [None]
            return rt[0], rt[1]
        else:
            return None, None

    def get_unmasked_feat(self, sp_seq, n_layer):
        """Returns unmasked features from n-th layer ConvBlock"""
        unmasked_feat = sp_seq.permute(0, 2, 1)  # BxTxC -> BxCxT
        for i in range(self.n_blocks):
            unmasked_feat = self.blocks[i](unmasked_feat)
            if i == n_layer:
                unmasked_feat = unmasked_feat.permute(0, 2, 1)
                break
        return unmasked_feat

    def forward(self, sp_seq, testing=False):
        # BxTxC -> BxCxT (reversed in Masked ConvBlock)
        unmasked_feat = sp_seq.permute(0, 2, 1)
        # Forward through each layer
        for i in range(self.n_blocks):
            unmasked_feat = self.blocks[i](unmasked_feat)
            if self.disable_cross_layer:
                # Last layer masked feature only
                if i == (self.n_blocks - 1):
                    feat = self.masked_convs[i](unmasked_feat)
            else:
                # Masked feature aggregation
                masked_feat = self.masked_convs[i](unmasked_feat)
                if i == 0:
                    feat = masked_feat
                else:
                    feat = feat + masked_feat
        # Apply bottleneck and predict spectrogram
        if self.apply_vq:
            q_feat = []
            offet = 0
            for vq_layer, cd in zip(self.vq_layers, self.vq_code_dims):
                _, q_f = vq_layer(feat[:, :, offet : offet + cd], testing)
                q_feat.append(q_f)
                offet += cd
            q_feat = torch.cat(q_feat, dim=-1)
            pred = self.postnet(q_feat)
        elif self.apply_ae:
            feat = self.ae_bottleneck(feat)
            pred = self.postnet(feat)
        else:
            pred = self.postnet(feat)
        return pred, feat
