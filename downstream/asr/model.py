import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def downsample(x, x_len, sample_rate, sample_style):
    batch_size, timestep, feature_dim = x.shape
    x_len = x_len // sample_rate

    if sample_style == 'drop':
        # Drop the unselected timesteps
        x = x[:, ::sample_rate, :].contiguous()
    elif sample_style == 'concat':
        # Drop the redundant frames and concat the rest according to sample rate
        if timestep % sample_rate != 0:
            x = x[:, :-(timestep % sample_rate), :]
        x = x.contiguous().view(batch_size, int(
            timestep / sample_rate), feature_dim * sample_rate)
    else:
        raise NotImplementedError
    
    return x, x_len


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


class RNNs(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        upstream_rate,
        module,
        bidirection,
        dim,
        dropout,
        layer_norm,
        proj,
        sample_rate,
        sample_style,
        total_rate = 320,
    ):
        super(RNNs, self).__init__()
        latest_size = input_size

        self.sample_rate = 1 if total_rate == -1 else round(total_rate / upstream_rate)
        self.sample_style = sample_style
        if sample_style == 'concat':
            latest_size *= self.sample_rate

        self.rnns = nn.ModuleList()
        for i in range(len(dim)):
            rnn_layer = RNNLayer(
                latest_size,
                module,
                bidirection,
                dim[i],
                dropout[i],
                layer_norm[i],
                sample_rate[i],
                proj[i],
            )
            self.rnns.append(rnn_layer)
            latest_size = rnn_layer.out_dim

        self.linear = nn.Linear(latest_size, output_size)
    
    def forward(self, x, x_len):
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, input_length, num_features).
            x_len (torch.IntTensor): Tensor of dimension (batch_size).
        Returns:
            Tensor: Predictor tensor of dimension (batch_size, input_length, number_of_classes).
        """
        # Perform Downsampling
        if self.sample_rate > 1:
            x, x_len = downsample(x, x_len, self.sample_rate, self.sample_style)

        for rnn in self.rnns:
            x, x_len = rnn(x, x_len)

        logits = self.linear(x)
        return logits, x_len        


class Wav2Letter(nn.Module):
    """
    The Wav2Letter model modified from torchaudio.models.Wav2Letter which preserves
    total downsample rate given the different upstream downsample rate.
    """

    def __init__(self, input_dim, output_dim, upstream_rate, total_rate=320, **kwargs):
        super(Wav2Letter, self).__init__()
        first_stride = 1 if total_rate == -1 else total_rate // upstream_rate
        self.downsample_rate = first_stride

        self.acoustic_model = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=250, kernel_size=48, stride=first_stride, padding=23),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=output_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_len):
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, input_length, num_features).
            x_len (torch.IntTensor): Tensor of dimension (batch_size).
        Returns:
            Tensor: Predictor tensor of dimension (batch_size, input_length, number_of_classes).
        """
        x = self.acoustic_model(x.transpose(1, 2).contiguous())
        return x.transpose(1, 2).contiguous(), x_len // self.downsample_rate
