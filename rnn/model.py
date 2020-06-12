# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ rnn/model.py ]
#   Synopsis     [ implementation of the rnn models, for now including `apc` ]
#   Author       [ Yu-An Chung ]
#   Copyright    [ https://github.com/iamyuanchung/Autoregressive-Predictive-Coding ]
#   Reference    [ https://arxiv.org/abs/1904.03240 ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Prenet(nn.Module):
    """Prenet is a multi-layer fully-connected network with ReLU activations.
    During training and testing (feature extraction), each input frame is passed
    into the Prenet, and the Prenet output is fed to the RNN.

    If no Prenet configuration is given, the input frames will be directly fed to
    the RNN without any transformation.
    """

    def __init__(self, input_size, num_layers, hidden_size, dropout):
        super(Prenet, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        # Don't get confused by the conv operation here -- since kernel_size and stride
        # are both 1, the operation here is equivalent to a fully-connected network.
        self.layers = nn.ModuleList(
            [nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1)
            for (in_size, out_size) in zip(input_sizes, output_sizes)])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, inputs):
        # inputs: (batch_size, seq_len, mel_dim)
        inputs = torch.transpose(inputs, 1, 2)
        # inputs: (batch_size, mel_dim, seq_len) -- for conv1d operation

        for layer in self.layers:
            inputs = self.dropout(self.relu(layer(inputs)))
        # inputs: (batch_size, last_dim, seq_len)

        return torch.transpose(inputs, 1, 2)
        # inputs: (batch_size, seq_len, last_dim) -- back to the original shape


class Postnet(nn.Module):
    """Postnet is a simple linear layer for predicting the target frames given the
    RNN context during training. We don't need the Postnet for feature extraction.
    """

    def __init__(self, input_size, output_size=80):
        super(Postnet, self).__init__()
        self.layer = nn.Conv1d(
            in_channels=input_size, out_channels=output_size, kernel_size=1, stride=1)


    def forward(self, inputs):
        # inputs: (batch_size, seq_len, hidden_size)
        inputs = torch.transpose(inputs, 1, 2)
        # inputs: (batch_size, hidden_size, seq_len) -- for conv1d operation

        return torch.transpose(self.layer(inputs), 1, 2)
        # (batch_size, seq_len, output_size) -- back to the original shape


class APCModel(nn.Module):
    """This class defines Autoregressive Predictive Coding (APC), a model that
    learns to extract general speech features from unlabeled speech data. These
    features are shown to contain rich speaker and phone information, and are
    useful for a wide range of downstream tasks such as speaker verification
    and phone classification.

    An APC model consists of a Prenet (optional), a multi-layer GRU network,
    and a Postnet. For each time step during training, the Prenet transforms
    the input frame into a latent representation, which is then consumed by
    the GRU network for generating internal representations across the layers.
    Finally, the Postnet takes the output of the last GRU layer and attempts to
    predict the target frame.

    After training, to extract features from the data of your interest, which
    do not have to be i.i.d. with the training data, simply feed-forward the
    the data through the APC model, and take the the internal representations
    (i.e., the GRU hidden states) as the extracted features and use them in
    your tasks.
    """

    def __init__(self, mel_dim, prenet_config, rnn_config):
        super(APCModel, self).__init__()
        self.mel_dim = mel_dim

        if prenet_config is not None:
            # Make sure the dimensionalities are correct
            assert prenet_config.input_size == mel_dim
            assert prenet_config.hidden_size == rnn_config.input_size
            assert rnn_config.input_size == rnn_config.hidden_size
            self.prenet = Prenet(
                input_size=prenet_config.input_size,
                num_layers=prenet_config.num_layers,
                hidden_size=prenet_config.hidden_size,
                dropout=prenet_config.dropout)
        else:
            assert rnn_config.input_size == mel_dim
            self.prenet = None

        in_sizes = [rnn_config.input_size] + [rnn_config.hidden_size] * (rnn_config.num_layers - 1)
        out_sizes = [rnn_config.hidden_size] * rnn_config.num_layers
        self.rnns = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True)
            for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.rnn_dropout = nn.Dropout(rnn_config.dropout)
        self.rnn_residual = rnn_config.residual

        self.postnet = Postnet(
            input_size=rnn_config.hidden_size,
            output_size=self.mel_dim)


    def forward(self, inputs, lengths):
        """Forward function for both training and testing (feature extraction).

        input:
            inputs: (batch_size, seq_len, mel_dim)
            lengths: (batch_size,)

        return:
            predicted_mel: (batch_size, seq_len, mel_dim)
            internal_reps: (num_layers + x, batch_size, seq_len, rnn_hidden_size),
                where x is 1 if there's a prenet, otherwise 0
        """
        seq_len = inputs.size(1)

        if self.prenet is not None:
            rnn_inputs = self.prenet(inputs)
            # rnn_inputs: (batch_size, seq_len, rnn_input_size)
            internal_reps = [rnn_inputs]
            # also include prenet_outputs in internal_reps
        else:
            rnn_inputs = inputs
            internal_reps = []

        packed_rnn_inputs = pack_padded_sequence(rnn_inputs, lengths, True)

        for i, layer in enumerate(self.rnns):
            packed_rnn_outputs, _ = layer(packed_rnn_inputs)

            rnn_outputs, _ = pad_packed_sequence(
                packed_rnn_outputs, True, total_length=seq_len)
            # outputs: (batch_size, seq_len, rnn_hidden_size)

            if i + 1 < len(self.rnns):
                # apply dropout except the last rnn layer
                rnn_outputs = self.rnn_dropout(rnn_outputs)

            rnn_inputs, _ = pad_packed_sequence(
                packed_rnn_inputs, True, total_length=seq_len)
            # rnn_inputs: (batch_size, seq_len, rnn_hidden_size)

            if self.rnn_residual and rnn_inputs.size(-1) == rnn_outputs.size(-1):
                # Residual connections
                rnn_outputs = rnn_outputs + rnn_inputs

            internal_reps.append(rnn_outputs)

            packed_rnn_inputs = pack_padded_sequence(rnn_outputs, lengths, True)

        predicted_mel = self.postnet(rnn_outputs)
        # predicted_mel: (batch_size, seq_len, mel_dim)

        internal_reps = torch.stack(internal_reps)

        return predicted_mel, internal_reps
        # predicted_mel is only for training; internal_reps is the extracted features
