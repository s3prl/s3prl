# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ loss.py ]
#   Synopsis     [ the objective functions for speech separation ]
#   Source       [ Use some code from https://github.com/funcwj/uPIT-for-speech-separation and https://github.com/asteroid-team/asteroid ]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""

import torch
from itertools import permutations
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from asteroid.losses import PITLossWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MSELoss(object):
    def __init__(self, num_srcs, mask_type):
        """
        Args:
            num_srcs (int):
                number of sources

            mask_type (str):
                type of mask to approach, currently supporting AM, PSM and
                NPSM. Please see Kolbæk M, Yu D, Tan Z H, et al
                Multitalker speech separation with utterance-level permutation
                invariant training of deep recurrent neural network
                for details
        """
        self.num_srcs = num_srcs
        self.mask_type = mask_type
        assert self.mask_type in ["AM", "PSM", "NPSM"]

    def compute_loss(self, masks, feat_length, source_attr, target_attr):
        feat_length = feat_length.to(device)
        mixture_spect = source_attr["magnitude"].to(device)
        targets_spect = [t.to(device) for t in target_attr["magnitude"]]
        mixture_phase = source_attr["phase"].to(device)
        targets_phase = [t.to(device) for t in target_attr["phase"]]

        def loss(permute):
            loss_for_permute = []
            for s, t in enumerate(permute):
                if self.mask_type == "AM":
                    refer_spect = targets_spect[t]
                elif self.mask_type == "PSM":
                    refer_spect = targets_spect[t] * torch.cos(mixture_phase - targets_phase[t])
                elif self.mask_type == "NPSM":
                    refer_spect = targets_spect[t] * F.relu(torch.cos(mixture_phase - targets_phase[t]))
                else:
                    raise ValueError("Mask type not defined.")

                utt_loss = torch.sum(
                    torch.sum(
                        torch.pow(masks[s] * mixture_spect - refer_spect, 2), -1),
                    -1)
                loss_for_permute.append(utt_loss)
            loss_perutt = sum(loss_for_permute) / feat_length
            return loss_perutt

        num_utts = feat_length.shape[0]
        pscore = torch.stack(
            [loss(p) for p in permutations(range(self.num_srcs))])
        min_perutt, _ = torch.min(pscore, dim=0)
        return torch.sum(min_perutt) / (self.num_srcs * num_utts)

class SISDRLoss(object):
    def __init__(self, num_srcs, n_fft, hop_length, win_length, window, center):
        self.num_srcs = num_srcs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        if window == 'hann':
            self.window = torch.hann_window(win_length).cuda()
        self.center = center
        self.loss = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx")

    def compute_loss(self, masks, feat_length, source_attr, wav_length, target_wav_list):
        mixture_stft = source_attr["stft"].to(device)
        bs = mixture_stft.size(0)
        est_targets = torch.zeros(bs, self.num_srcs, max(wav_length)).to(device)
        targets = torch.stack(target_wav_list, dim=1).to(device)
        for i in range(bs):
            mix_stft_utt = mixture_stft[i, :feat_length[i], :]
            est_src_list = []
            for j in range(self.num_srcs):
                mask_utt = masks[j][i, :feat_length[i], :]
                est_stft_utt = mix_stft_utt * mask_utt
                est_stft_utt = (torch.transpose(est_stft_utt, 0, 1)).unsqueeze(0)
                est_src = torch.istft(est_stft_utt, self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=self.center, length=wav_length[i])[0]
                if est_src.size(0) != wav_length[i]:
                    print("Warning: wav length doesn't match")
                    est_src = match_wave_length(est_src, wav_length[i])
                est_src_list.append(est_src)
            est_srcs = torch.stack(est_src_list, dim=0)
            est_targets[i, :, :wav_length[i]] = est_srcs
        loss = self.loss(est_targets, targets, length=wav_length)
        return loss


class PairwiseNegSDR(_Loss):
    r"""Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target
            and estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.

    Shape:
        - est_targets : :math:`(batch, nsrc, ...)`.
        - targets: :math:`(batch, nsrc, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
        >>>                            pit_from='pairwise')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super(PairwiseNegSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_targets, targets, length):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        assert targets.size() == est_targets.size()
        length = length.to(device)
        mask = length_mask(length).to(device)
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.sum(targets, dim=2, keepdim=True) / length.view(-1, 1, 1)
            mean_estimate = torch.sum(est_targets, dim=2, keepdim=True) / length.view(-1, 1, 1)
            targets = (targets - mean_source) * torch.unsqueeze(mask, 1)
            est_targets = (est_targets - mean_estimate) * torch.unsqueeze(mask, 1)
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)

        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -pair_wise_sdr

def match_wave_length(x, length):
    if x.size(0) == length:
        return x
    elif x.size(0) > length:
        new_x = x[:length]
        return new_x
    else:
        new_x = torch.zeros(length).to(x.device)
        new_x[:x.size(0)] = x
        return new_x

def length_mask(length):
    mask = torch.zeros(len(length), max(length)).to(device)
    for i in range(len(length)):
        mask[i, :length[i]] = 1
    return mask
