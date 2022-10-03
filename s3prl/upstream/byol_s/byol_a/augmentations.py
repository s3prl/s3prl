"""BYOL for Audio: Augmentation modules.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SpecAugment(nn.Module):
    """
    SpecAugment module, without time warping.

    Original paper: 1904.08779

    "Time warping contributes, but is not a major factor in improving performance."
    """

    def __init__(self, pF=0.1, mF=2, pT=0.1, mT=2):
        super().__init__()
        self.pF = pF
        self.mF = mF
        self.pT = pT
        self.mT = mT

    def forward(self, x):
        _, n_mels, n_steps = x.shape

        mask_value = x.mean()

        # Frequency masking
        freq_mask_param = self.pF * n_mels
        for _ in range(self.mF):
            x = torchaudio.transforms.FrequencyMasking(freq_mask_param)(x, mask_value)

        # Time masking
        time_mask_param = self.pT * n_steps
        for _ in range(self.mT):
            x = torchaudio.transforms.TimeMasking(time_mask_param)(x, mask_value)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(pF={self.pF}, mF={self.mF}, pT={self.pT}, mT={self.mT})"


class TimeFrequencyMasking(nn.Module):
    """Time-frequency masking option inspired by: https://arxiv.org/abs/2102.01243

    Attributes
    ----------
    freq_mask_param: int, default=48
        maximum possible length of the mask. Indices uniformly sampled from [0, freq_mask_param)
    time_mask_param: int, default=192
        maximum possible length of the mask. Indices uniformly sampled from [0, time_mask_param)
    """

    def __init__(self, freq_mask_param=48, time_mask_param=192):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def forward(self, x):
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(F={self.freq_mask_param}, T={self.time_mask_param})"


class RandomResizeCrop(nn.Module):
    """Random Resize Crop block.

    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(
        self,
        virtual_crop_scale=(1.0, 1.5),
        freq_scale=(0.6, 1.5),
        time_scale=(0.6, 1.5),
    ):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = "bicubic"
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms):
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [
            int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)
        ]
        virtual_crop_area = (
            torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
            .to(torch.float)
            .to(lms.device)
        )
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y : y + h, x : x + w] = lms
        # get random area
        i, j, h, w = self.get_params(
            virtual_crop_area.shape[-2:],
            lms.shape[-2:],
            self.time_scale,
            self.freq_scale,
        )
        crop = virtual_crop_area[:, i : i + h, j : j + w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = F.interpolate(
            crop.unsqueeze(0),
            size=lms.shape[-2:],
            mode=self.interpolation,
            align_corners=True,
        ).squeeze(0)
        return lms.to(torch.float)

    def __repr__(self):
        format_string = (
            self.__class__.__name__ + f"(virtual_crop_size={self.virtual_crop_scale}"
        )
        format_string += ", time_scale={0}".format(
            tuple(round(s, 4) for s in self.time_scale)
        )
        format_string += ", freq_scale={0})".format(
            tuple(round(r, 4) for r in self.freq_scale)
        )
        return format_string


def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1.0 - alpha) * xb
    return torch.log(x + torch.finfo(x.dtype).eps)


class MixupBYOLA(nn.Module):
    """Mixup for BYOL-A.

    Args:
        ratio: Alpha in the paper.
        n_memory: Size of memory bank FIFO.
        log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
    """

    def __init__(self, ratio=0.4, n_memory=2048, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    def forward(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            # mix them
            mixed = (
                log_mixup_exp(x, z, 1.0 - alpha)
                if self.log_mixup_exp
                else alpha * z + (1.0 - alpha) * x
            )
        else:
            mixed = x
        # update memory bank
        self.memory_bank = (self.memory_bank + [x])[-self.n :]

        return mixed.to(torch.float)

    def __repr__(self):
        format_string = self.__class__.__name__ + f"(ratio={self.ratio},n={self.n}"
        format_string += f",log_mixup_exp={self.log_mixup_exp})"
        return format_string


class MixGaussianNoise:
    """Gaussian Noise Mixer.
    This interpolates with random sample, unlike Mixup.
    """

    def __init__(self, ratio=0.3):
        self.ratio = ratio

    def forward(self, lms):
        x = lms.exp()

        lambd = self.ratio * np.random.rand()
        z = torch.normal(0, lambd, x.shape).exp()
        mixed = (1 - lambd) * x + z + torch.finfo(x.dtype).eps

        return mixed.log()

    def __repr__(self):
        format_string = self.__class__.__name__ + f"(ratio={self.ratio})"
        return format_string


class RunningMean:
    """Running mean calculator for arbitrary axis configuration."""

    def __init__(self, axis):
        self.n = 0
        self.axis = axis

    def put(self, x):
        # https://math.stackexchange.com/questions/106700/incremental-averageing
        if self.n == 0:
            self.mu = x.mean(self.axis, keepdims=True)
        else:
            self.mu += (x.mean(self.axis, keepdims=True) - self.mu) / self.n
        self.n += 1

    def __call__(self):
        return self.mu

    def __len__(self):
        return self.n


class RunningVariance:
    """Calculate mean/variance of tensors online.
    Thanks to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, axis, mean):
        self.update_mean(mean)
        self.s2 = RunningMean(axis)

    def update_mean(self, mean):
        self.mean = mean

    def put(self, x):
        self.s2.put((x - self.mean) ** 2)

    def __call__(self):
        return self.s2()

    def std(self):
        return np.sqrt(self())


class RunningNorm(nn.Module):
    """Online Normalization using Running Mean/Std.

    This module will only update the statistics up to the specified number of epochs.
    After the `max_update_epochs`, this will normalize with the last updated statistics.

    Args:
        epoch_samples: Number of samples in one epoch
        max_update_epochs: Number of epochs to allow update of running mean/variance.
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, epoch_samples, max_update_epochs=10, axis=[1, 2]):
        super().__init__()
        self.max_update = epoch_samples * max_update_epochs
        self.ema_mean = RunningMean(axis)
        self.ema_var = RunningVariance(axis, 0)

    def forward(self, image):
        if len(self.ema_mean) < self.max_update:
            self.ema_mean.put(image)
            self.ema_var.update_mean(self.ema_mean())
            self.ema_var.put(image)
            self.mean = self.ema_mean()
            self.std = torch.clamp(
                self.ema_var.std(), torch.finfo().eps, torch.finfo().max
            )
        return (image - self.mean) / self.std

    def __repr__(self):
        format_string = (
            self.__class__.__name__
            + f"(max_update={self.max_update},axis={self.ema_mean.axis})"
        )
        return format_string


class PrecomputedNorm(nn.Module):
    """Normalization using Pre-computed Mean/Std.

    Args:
        stats: Precomputed (mean, std).
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, stats, axis=[1, 2]):
        super().__init__()
        self.axis = axis
        self.mean, self.std = stats

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def __repr__(self):
        format_string = (
            self.__class__.__name__
            + f"(mean={self.mean}, std={self.std}, axis={self.axis})"
        )
        return format_string


class NormalizeBatch(nn.Module):
    """Normalization of Input Batch.

    Note:
        Unlike other blocks, use this with *batch inputs*.

    Args:
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, axis=[0, 2, 3]):
        super().__init__()
        self.axis = axis

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _mean = X.mean(dim=self.axis, keepdims=True)
        _std = torch.clamp(
            X.std(dim=self.axis, keepdims=True), torch.finfo().eps, torch.finfo().max
        )
        return (X - _mean) / _std

    def __repr__(self):
        format_string = self.__class__.__name__ + f"(axis={self.axis})"
        return format_string
