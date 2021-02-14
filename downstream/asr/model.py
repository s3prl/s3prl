import torch
import torch.nn as nn


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
