# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the simple (LSTMP), simple-AR and taco2-AR models for any-to-any voice conversion ]
#   Reference    [ `WaveNet Vocoder with Limited Training Data for Voice Conversion`, Interspeech 2018 ]
#   Reference    [ `Non-Parallel Voice Conversion with Autoregressive Conversion Model and Duration Adjustment`, Joint WS for BC & VCC 2020 ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

################################################################################

# The follow section is related to Tacotron2
# Reference: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/tacotron2

def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))

class Taco2Encoder(torch.nn.Module):
    """Encoder module of the Tacotron2 TTS model.

    Reference:
    _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        idim,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_chans=512,
        econv_filts=5,
        use_batch_norm=True,
        use_residual=False,
        dropout_rate=0.5,
    ):
        """Initialize Tacotron2 encoder module.

        Args:
            idim (int) Dimension of the inputs.
            elayers (int, optional) The number of encoder blstm layers.
            eunits (int, optional) The number of encoder blstm units.
            econv_layers (int, optional) The number of encoder conv layers.
            econv_filts (int, optional) The number of encoder conv filter size.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            use_residual (bool, optional) Whether to use residual connection.
            dropout_rate (float, optional) Dropout rate.

        """
        super(Taco2Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.use_residual = use_residual

        # define network layer modules
        self.input_layer = torch.nn.Linear(idim, econv_chans)

        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in range(econv_layers):
                ichans = econv_chans
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of the padded acoustic feature sequence (B, Lmax, idim)
        """
        xs = self.input_layer(xs).transpose(1, 2)

        if self.convs is not None:
            for i in range(len(self.convs)):
                if self.use_residual:
                    xs += self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)
        if self.blstm is None:
            return xs.transpose(1, 2)
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens.cpu(), batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Lmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens

class Taco2Prenet(torch.nn.Module):
    """Prenet module for decoder of Tacotron2.

    The Prenet preforms nonlinear conversion
    of inputs before input to auto-regressive lstm,
    which helps alleviate the exposure bias problem.

    Note:
        This module alway applies dropout even in evaluation.
        See the detail in `Natural TTS Synthesis by
        Conditioning WaveNet on Mel Spectrogram Predictions`_.

    _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_
       https://arxiv.org/abs/1712.05884

    """

    def __init__(self, idim, n_layers=2, n_units=256, dropout_rate=0.5):
        super(Taco2Prenet, self).__init__()
        self.dropout_rate = dropout_rate
        self.prenet = torch.nn.ModuleList()
        for layer in range(n_layers):
            n_inputs = idim if layer == 0 else n_units
            self.prenet += [
                torch.nn.Sequential(torch.nn.Linear(n_inputs, n_units), torch.nn.ReLU())
            ]

    def forward(self, x):
        # Make sure at least one dropout is applied.
        if len(self.prenet) == 0:
            return F.dropout(x, self.dropout_rate)

        for i in range(len(self.prenet)):
            x = F.dropout(self.prenet[i](x), self.dropout_rate)
        return x

################################################################################

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''

    def __init__(self, input_dim, module, bidirection, dim, dropout, layer_norm, sample_rate, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.proj = proj

        # Recurrent layer
        self.layer = getattr(nn, module.upper())(
            input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):

        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()

        input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.layer(input_x)
        output, x_len = pad_packed_sequence(output, batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            output, x_len = downsample(output, x_len, self.sample_rate, 'drop')

        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len


class RNNCell(nn.Module):
    ''' RNN cell wrapper'''

    def __init__(self, input_dim, module, dim, dropout, layer_norm, proj):
        super(RNNCell, self).__init__()
        # Setup
        rnn_out_dim = dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.proj = proj

        # Recurrent cell
        self.cell = getattr(nn, module.upper()+"Cell")(input_dim, dim)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)
    
    def forward(self, input_x, z, c):

        # Forward RNN cell
        new_z, new_c = self.cell(input_x, (z, c))

        # Normalizations
        if self.layer_norm:
            new_z = self.ln(new_z)
        if self.dropout > 0:
            new_z = self.dp(new_z)

        if self.proj:
            new_z = torch.tanh(self.pj(new_z))

        return new_z, new_c

################################################################################

class Model(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 resample_ratio,
                 stats,
                 ar,
                 encoder_type,
                 hidden_dim,
                 lstmp_layers,
                 lstmp_dropout_rate,
                 lstmp_proj_dim,
                 lstmp_layernorm,
                 spk_emb_integration_type,
                 spk_emb_dim,
                 prenet_layers=2,
                 prenet_dim=256,
                 prenet_dropout_rate=0.5,
                 **kwargs):
        super(Model, self).__init__()

        self.ar = ar
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim # this is also the decoder output dim
        self.output_dim = output_dim # acoustic feature dim
        self.resample_ratio = resample_ratio
        self.spk_emb_integration_type = spk_emb_integration_type
        self.spk_emb_dim = spk_emb_dim
        if spk_emb_integration_type == "add":
            assert spk_emb_dim == hidden_dim

        self.register_buffer("target_mean", torch.from_numpy(stats.mean_).float())
        self.register_buffer("target_scale", torch.from_numpy(stats.scale_).float())

        # define encoder
        if encoder_type == "taco2":
            self.encoder = Taco2Encoder(input_dim, eunits=hidden_dim)
        elif encoder_type == "ffn":
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU()
            )
        else:
            raise ValueError("Encoder type not supported.")
        
        # define projection layer
        if self.spk_emb_integration_type == "add":
            self.spk_emb_projection = torch.nn.Linear(spk_emb_dim, hidden_dim)
        elif self.spk_emb_integration_type == "concat": 
            self.spk_emb_projection = torch.nn.Linear(
                hidden_dim + spk_emb_dim, hidden_dim
            )
        else:
            raise ValueError("Integration type not supported.")
        
        # define prenet
        self.prenet = Taco2Prenet(
            idim=output_dim,
            n_layers=prenet_layers,
            n_units=prenet_dim,
            dropout_rate=prenet_dropout_rate,
        )

        # define decoder (LSTMPs)
        self.lstmps = nn.ModuleList()
        for i in range(lstmp_layers):
            if ar:
                prev_dim = output_dim if prenet_layers == 0 else prenet_dim
                rnn_input_dim = hidden_dim + prev_dim if i == 0 else hidden_dim
                rnn_layer = RNNCell(
                    rnn_input_dim,
                    "LSTM",
                    hidden_dim,
                    lstmp_dropout_rate,
                    lstmp_layernorm,
                    proj=True,
                )
            else:
                rnn_input_dim = hidden_dim
                rnn_layer = RNNLayer(
                    rnn_input_dim,
                    "LSTM",
                    False,
                    hidden_dim,
                    lstmp_dropout_rate,
                    lstmp_layernorm,
                    sample_rate=1,
                    proj=True,
                )
            self.lstmps.append(rnn_layer)

        # projection layer
        self.proj = torch.nn.Linear(hidden_dim, output_dim)

    def normalize(self, x):
        return (x - self.target_mean) / self.target_scale
    
    def _integrate_with_spk_emb(self, hs, spembs):
        """Integrate speaker embedding with hidden states.
            Args:
                hs (Tensor): Batch of hidden state sequences (B, Lmax, hdim).
                spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).
        """
        if self.spk_emb_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.spk_emb_projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_emb_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.spk_emb_projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def forward(self, features, lens, ref_spk_embs, targets = None):
        """Calculate forward propagation.
            Args:
            features: Batch of the sequences of input features (B, Lmax, idim).
            targets: Batch of the sequences of padded target features (B, Lmax, odim).
            ref_spk_embs: Batch of the sequences of reference speaker embeddings (B, spk_emb_dim).
        """
        B = features.shape[0]
        
        # resample the input features according to resample_ratio
        features = features.permute(0, 2, 1)
        resampled_features = F.interpolate(features, scale_factor = self.resample_ratio)
        resampled_features = resampled_features.permute(0, 2, 1)
        lens = lens * self.resample_ratio

        # encoder
        if self.encoder_type == "taco2":
            encoder_states, lens = self.encoder(resampled_features, lens) # (B, Lmax, hidden_dim)
        elif self.encoder_type == "ffn":
            encoder_states = self.encoder(resampled_features) # (B, Lmax, hidden_dim)

        # inject speaker embeddings
        encoder_states = self._integrate_with_spk_emb(encoder_states, ref_spk_embs)
        
        # decoder: LSTMP layers & projection
        if self.ar:
            if targets is not None:
                targets = targets.transpose(0, 1) # (Lmax, B, output_dim)
            predicted_list = []

            # initialize hidden states
            c_list = [encoder_states.new_zeros(B, self.hidden_dim)]
            z_list = [encoder_states.new_zeros(B, self.hidden_dim)]
            for _ in range(1, len(self.lstmps)):
                c_list += [encoder_states.new_zeros(B, self.hidden_dim)]
                z_list += [encoder_states.new_zeros(B, self.hidden_dim)]
            prev_out = encoder_states.new_zeros(B, self.output_dim)

            # step-by-step loop for autoregressive decoding
            for t, encoder_state in enumerate(encoder_states.transpose(0, 1)):
                concat = torch.cat([encoder_state, self.prenet(prev_out)], dim=1) # each encoder_state has shape (B, hidden_dim)
                for i, lstmp in enumerate(self.lstmps):
                    lstmp_input = concat if i == 0 else z_list[i-1]
                    z_list[i], c_list[i] = lstmp(lstmp_input, z_list[i], c_list[i])
                predicted_list += [self.proj(z_list[-1]).view(B, self.output_dim, -1)] # projection is done here to ensure output dim
                prev_out = targets[t] if targets is not None else predicted_list[-1].squeeze(-1) # targets not None = teacher-forcing
                prev_out = self.normalize(prev_out) # apply normalization
            predicted = torch.cat(predicted_list, dim=2)
            predicted = predicted.transpose(1, 2)  # (B, hidden_dim, Lmax) -> (B, Lmax, hidden_dim)
        else:
            predicted = encoder_states
            for i, lstmp in enumerate(self.lstmps):
                predicted, lens = lstmp(predicted, lens)
        
            # projection layer
            predicted = self.proj(predicted)

        return predicted, lens
