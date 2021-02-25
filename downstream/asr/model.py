import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMs(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        upstream_rate,
        hidden_size = 256,
        n_layers = 3,
        dropout = 0.2,
        bidirectional = True,
        total_rate = 320,
        sample_style = 'concat'
    ):
        super(LSTMs, self).__init__()
        latest_size = input_size

        self.sample_rate = 1 if total_rate == -1 else round(total_rate / upstream_rate)
        self.sample_style = sample_style
        if sample_style == 'concat':
            latest_size *= self.sample_rate

        self.model = nn.LSTM(
            input_size = latest_size,
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout,
            bidirectional = bidirectional,
        )
        latest_size = hidden_size
        if bidirectional:
            latest_size *= 2

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
            batch_size, timestep, x_size = x.shape
            x_len = x_len // self.sample_rate

            if self.sample_style == 'drop':
                # Drop the unselected timesteps
                output = output[:, ::self.sample_rate, :].contiguous()
            elif self.sample_style == 'concat':
                # Drop the redundant frames and concat the rest according to sample rate
                if timestep % self.sample_rate != 0:
                    x = x[:, :-(timestep % self.sample_rate), :]
                x = x.contiguous().view(batch_size, int(
                    timestep / self.sample_rate), x_size * self.sample_rate)
            else:
                raise NotImplementedError

        packed_x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.model(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        logits = self.linear(out)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        return log_probs, x_len        


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
        x = nn.functional.log_softmax(x, dim=1)
        return x.transpose(1, 2).contiguous(), x_len // self.downsample_rate
