import torch
import torch.nn as nn
import torchaudio
from torch.nn.functional import conv1d, conv2d

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization


class AugmentMelSTFT(nn.Module):
    def __init__(
        self,
        n_mels=128,
        sr=32000,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        freqm=48,
        timem=192,
        htk=False,
        fmin=0.0,
        fmax=None,
        norm=1,
        fmin_aug_range=1,
        fmax_aug_range=1000,
    ):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin

        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2

        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=False), persistent=False
        )
        assert (
            fmin_aug_range >= 1
        ), f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert (
            fmin_aug_range >= 1
        ), f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer(
            "preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False
        )
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):

        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(
            1
        )
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
        )
        x = (x**2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = (
            self.fmax
            + self.fmax_aug_range // 2
            - torch.randint(self.fmax_aug_range, (1,)).item()
        )
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sr,
            fmin,
            fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0),
            device=x.device,
        )
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec

    def extra_repr(self):
        return "winsize={}, hopsize={}".format(self.win_length, self.hopsize)
