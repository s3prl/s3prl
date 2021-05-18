# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol/model.py ]
#   Synopsis     [ Implementation of the SpeechNet Audio Encoder model ]
#   Author       [ Chen, Yi Chen (https://github.com/grtzsohalf) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/grtzsohalf/SpeechNet/blob/master/src/audio_encoder.py ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn

from upstream.byol.module import VGGExtractor, CNNExtractor, PseudoDownsampler, RNNLayer

from upstream.byol.conformer import ConvolutionModule
from upstream.byol.conformer import EncoderLayer
from upstream.byol.conformer import Swish

from upstream.byol.transformer import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from upstream.byol.transformer import (
    PositionalEncoding,
    ScaledPositionalEncoding,
    RelPositionalEncoding,
)
from upstream.byol.transformer import LayerNorm
from upstream.byol.transformer import PositionwiseFeedForward
from upstream.byol.transformer import repeat
from upstream.byol.transformer import make_non_pad_mask
from upstream.byol.sap import SAP

import torch.cuda.nvtx as nvtx


def get_activation(act):
    """Return activation function."""

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
        "gelu": torch.nn.GELU,
    }

    return activation_funcs[act]()


class AudioEncoder(nn.Module):
    ''' Encoder (a.k.a. Listener in LAS)
        Encodes acoustic feature to latent representation, see config file for more details.'''

    def __init__(self, input_size, prenet, module, dim, dropout, 
                 bidirection=None, layer_norm=None, proj=None, sample_rate=None, sample_style=None,
                 layer=None, layer_share=None, layer_content=None, layer_speaker=None,
                 head=None, linear_unit=None, normalized_before=None, concat_after=None,
                 macaron_style=False,
                 pos_enc_layer_type="abs_pos",
                 selfattention_layer_type="selfattn",
                 use_cnn_module=False,
                 cnn_activation_type="swish",
                 cnn_module_kernel=31,
                 process_group=None,
                 pretrain=False,
                 ):
        super(AudioEncoder, self).__init__()

        self.input_size = input_size
        self.pretrain = pretrain

        # Hyper-parameters checking
        self.vgg = prenet == 'vgg'
        self.cnn = prenet == 'cnn'
        self.sample_rate = 1
        if module in ['LSTM', 'GRU']:
            assert len(sample_rate) == len(dropout), 'Number of layer mismatch'
            assert len(dropout) == len(dim), 'Number of layer mismatch'
            num_layers = len(dim)
            assert num_layers >= 1, 'Encoder should have at least 1 layer'
        elif module == 'Transformer':
            self.layer = None

        # Construct model
        input_dim = input_size

        self.module = module
        self.dim = dim
        self.dropout = dropout

        # Transformer specific
        self.layer_share = layer_share
        self.layer_content = layer_content
        self.layer_speaker = layer_speaker
        self.head = head
        self.linear_unit = linear_unit
        self.normalized_before = normalized_before
        self.concat_after = concat_after

        # Prenet on audio feature
        if self.vgg:
            if module in ['LSTM', 'GRU']:
                content_vgg_extractor = VGGExtractor(input_size)
                #speaker_vgg_extractor = VGGExtractor(input_size)
            elif module == 'Transformer':
                content_vgg_extractor = VGGExtractor(input_size, hide_dim=dim)
                #speaker_vgg_extractor = VGGExtractor(input_size, hide_dim=dim)
            input_dim = content_vgg_extractor.out_dim
            self.sample_rate = self.sample_rate*4
            self.content_extractor = content_vgg_extractor
            #self.speaker_extractor = speaker_vgg_extractor
        elif self.cnn:
            if module in ['LSTM', 'GRU']:
                content_cnn_extractor = CNNExtractor(input_size, out_dim=dim[0])
                #speaker_cnn_extractor = CNNExtractor(input_size, out_dim=dim[0])
            elif module == 'Transformer':
                content_cnn_extractor = CNNExtractor(input_size, out_dim=dim)
                #speaker_cnn_extractor = CNNExtractor(input_size, out_dim=dim)
            input_dim = content_cnn_extractor.out_dim
            self.sample_rate = self.sample_rate*4
            self.content_extractor = content_cnn_extractor
            #self.speaker_extractor = speaker_cnn_extractor
        else:
            self.content_extractor = PseudoDownsampler(input_size, dim)
            #self.speaker_extractor = PseudoDownsampler(input_size, dim)
        self.speaker_extractor = PseudoDownsampler(input_size, dim)

        # Recurrent or self-attention encoder
        if module in ['LSTM', 'GRU']:
            module_list = nn.ModuleList()
            for l in range(num_layers):
                module_list.append(RNNLayer(input_dim, module, dim[l], bidirection, dropout[l], layer_norm[l],
                                            sample_rate[l], sample_style, proj[l]))
            self.encoder_layers = nn.ModuleList(module_list)
            input_dim = module_list[-1].out_dim
            self.sample_rate = self.sample_rate*sample_rate[l]
        elif module == 'Transformer':
            # get positional embedding class
            if pos_enc_layer_type == "abs_pos":
                pos_enc_class = PositionalEncoding
            elif pos_enc_layer_type == "scaled_abs_pos":
                pos_enc_class = ScaledPositionalEncoding
            elif pos_enc_layer_type == "rel_pos":
                assert selfattention_layer_type == "rel_selfattn"
                pos_enc_class = RelPositionalEncoding
            else:
                pos_enc_class = None

            # get positional embedding module
            self.pos_embedding = pos_enc_class(dim, dropout) if pos_enc_class else None

            # self-attention module definition
            if selfattention_layer_type == "selfattn":
                encoder_selfattn_layer = MultiHeadedAttention
                encoder_selfattn_layer_args = (head, dim, dropout)
            elif selfattention_layer_type == "rel_selfattn":
                assert pos_enc_layer_type == "rel_pos"
                encoder_selfattn_layer = RelPositionMultiHeadedAttention
                encoder_selfattn_layer_args = (head, dim, dropout)
            else:
                raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

            # get activation class
            cnn_activation = get_activation(cnn_activation_type)

            # Content layers in Encoder
            if layer_content > 0:
                assert layer_speaker == 0
                self.content_encoder = repeat(
                    layer_content,
                    lambda lnum: EncoderLayer(
                        dim,
                        encoder_selfattn_layer(*encoder_selfattn_layer_args),
                        PositionwiseFeedForward(dim, linear_unit, dropout),
                        PositionwiseFeedForward(dim, linear_unit, dropout) if macaron_style else None,
                        ConvolutionModule(process_group, dim, cnn_module_kernel, cnn_activation) if use_cnn_module else None,
                        dropout,
                        normalized_before,
                        concat_after,
                    ),
                )

            # Speaker layers in Encoder
            if layer_speaker > 0:
                assert layer_content == 0
                self.speaker_encoder = repeat(
                    layer_speaker,
                    lambda lnum: EncoderLayer(
                        dim,
                        encoder_selfattn_layer(*encoder_selfattn_layer_args),
                        PositionwiseFeedForward(dim, linear_unit, dropout),
                        PositionwiseFeedForward(dim, linear_unit, dropout) if macaron_style else None,
                        ConvolutionModule(process_group, dim, cnn_module_kernel, cnn_activation) if use_cnn_module else None,
                        dropout,
                        normalized_before,
                        concat_after,
                    ),
                )
                self.speaker_aggregater = SAP(dim)

        else:
            raise NotImplementedError

        # Build model
        self.in_dim = input_size
        self.out_dim = input_dim

    def forward(self, input_x, enc_len=None):
        if enc_len is None:
            enc_len =  (input_x.sum(dim=-1) != 0).long().sum(dim=-1) # (B, T, F) -> (B)

        nvtx.range_push('Audio encoder extract')
        content_input_x, content_enc_len = self.content_extractor(input_x, enc_len)
        speaker_input_x, speaker_enc_len = self.speaker_extractor(input_x, enc_len)
        nvtx.range_pop()

        if self.module in ['LSTM', 'GRU']:
            for _, layer in enumerate(self.encoder_layers):
                content_input_x, content_enc_len = layer(content_input_x, content_enc_len)
            content_enc_mask = None
            speaker_input_x = None
            speaker_enc_len = None
            speaker_enc_mask = None

        elif self.module == 'Transformer':
            nvtx.range_push('Make non pad mask')
            content_enc_mask = make_non_pad_mask(content_enc_len.tolist()).to(content_input_x.device).unsqueeze(-2)
            speaker_enc_mask = make_non_pad_mask(speaker_enc_len.tolist()).to(speaker_input_x.device).unsqueeze(-2)
            nvtx.range_pop()

            if self.pos_embedding:
                nvtx.range_push('Pos embedding')
                content_input_x = self.pos_embedding(content_input_x)
                speaker_input_x = self.pos_embedding(speaker_input_x)
                nvtx.range_pop()

            # Content layers in Encoder
            if self.layer_content > 0:
                nvtx.range_push('Content encoder forward')
                content_input_x, content_enc_mask = self.content_encoder(content_input_x, content_enc_mask)
                if isinstance(content_input_x, tuple):
                    content_input_x = content_input_x[0]
                nvtx.range_pop()
            else:
                content_input_x = None
                content_enc_mask = None
            
            # Speaker layers in Encoder
            if self.layer_speaker > 0:
                nvtx.range_push('Speaker encoder forward')
                speaker_input_x, speaker_enc_mask = self.speaker_encoder(speaker_input_x, speaker_enc_mask)
                if isinstance(speaker_input_x, tuple):
                    speaker_input_x = speaker_input_x[0]
                nvtx.range_pop()

                # attention pooling
                nvtx.range_push('Attention pooling')
                sap_input_enc_mask = ((1 - speaker_enc_mask.long()) * -10000.00).squeeze(1)
                speaker_input_x, sap_input_enc_mask = self.speaker_aggregater(speaker_input_x, sap_input_enc_mask)
                nvtx.range_pop()
            else:
                speaker_input_x = None
                speaker_enc_mask = None
        
        if self.pretrain and self.layer_content > 0:
            return torch.mean(content_input_x, axis=1)
        elif self.pretrain and self.layer_speaker > 0:
            return torch.mean(speaker_input_x, axis=1)
        else:
            return content_enc_len, content_input_x, content_enc_mask, speaker_input_x, speaker_enc_mask