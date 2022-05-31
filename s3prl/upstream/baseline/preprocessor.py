# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/baseline/preprocessor.py ]
#   Synopsis     [ the baseline feature preprocessor with torchaudio.transforms as backend ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


from functools import partial

###############
# IMPORTATION #
###############
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import compute_deltas
from torchaudio.transforms import MFCC, MelScale, Spectrogram

############
# CONSTANT #
############
N_SAMPLED_PSEUDO_WAV = 2


def get_preprocessor(audio_config, take_first_channel=True, process_input_only=False):
    """
    Args:
        take_first_channel:
            bool
            If True, the preprocessor takes input as: (*, channel=1, waveform_len),
                where `input` and `target` are taken from the same channel (i.e. the same and single waveform).
            If False, the preprocessor takes input as: (*, channel=2, waveform_len),
                where `input` and `target` are taken from the specified channel (i.e. multiple correlated views of the same waveform).
                For example: preprocessor_input = torch.cat([wav_aug1.unsqueeze(1), wav_aug2.unsqueeze(1)], dim=1) # shape: (batch_size, channel=2, seq_len)
        process_input_only:
            bool
            If True, the preprocessor will process `input`
            If False, the preprocessor will process both `input` and `target`
    """
    assert not (take_first_channel == False and process_input_only == True)

    if not "target" in audio_config:
        audio_config["target"] = audio_config["input"]

    input_feat = audio_config["input"]
    target_feat = audio_config["target"]
    if take_first_channel:
        input_feat["channel"] = 0
        target_feat["channel"] = 0

    preprocessor = OnlinePreprocessor(
        **audio_config, feat_list=[input_feat, target_feat]
    )
    input_dim, target_dim = [feat.size(-1) for feat in preprocessor()]

    if process_input_only:
        del preprocessor
        preprocessor = OnlinePreprocessor(**audio_config, feat_list=[input_feat])
    return preprocessor, input_dim, target_dim


class OnlinePreprocessor(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        win_ms=25,
        hop_ms=10,
        n_freq=201,
        n_mels=40,
        n_mfcc=13,
        feat_list=None,
        eps=1e-10,
        **kwargs
    ):
        super(OnlinePreprocessor, self).__init__()
        # save preprocessing arguments
        self._sample_rate = sample_rate
        self._win_ms = win_ms
        self._hop_ms = hop_ms
        self._n_freq = n_freq
        self._n_mels = n_mels
        self._n_mfcc = n_mfcc

        win = round(win_ms * sample_rate / 1000)
        hop = round(hop_ms * sample_rate / 1000)
        n_fft = (n_freq - 1) * 2
        self._win_args = {"n_fft": n_fft, "hop_length": hop, "win_length": win}
        self.register_buffer("_window", torch.hann_window(win))

        self._stft_args = {
            "center": True,
            "pad_mode": "reflect",
            "normalized": False,
            "onesided": True,
        }
        # stft_args: same default values as torchaudio.transforms.Spectrogram & librosa.core.spectrum._spectrogram
        self._stft = partial(torch.stft, **self._win_args, **self._stft_args)
        self._melscale = MelScale(sample_rate=sample_rate, n_mels=n_mels)
        self._mfcc_trans = MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=self._win_args,
        )
        self._istft = partial(torch.istft, **self._win_args, **self._stft_args)

        self.feat_list = feat_list
        self.register_buffer(
            "_pseudo_wavs", torch.randn(N_SAMPLED_PSEUDO_WAV, sample_rate)
        )
        self.eps = eps

    def _magphase(self, spectrogram):
        # Replace the deprecated private member function self._magphase
        # Ref. https://github.com/pytorch/audio/issues/1337
        # original: `self._magphase = partial(torchaudio.functional.magphase, power=2)`
        spectrogram = torch.view_as_complex(spectrogram)
        return spectrogram.abs().pow(exponent=2), spectrogram.angle()

    def _check_list(self, feat_list):
        if feat_list is None:
            feat_list = self.feat_list
        assert type(feat_list) is list
        return feat_list

    def _transpose_list(self, feats):
        return [
            feat.transpose(-1, -2).contiguous() if type(feat) is torch.Tensor else feat
            for feat in feats
        ]

    @classmethod
    def get_feat_config(cls, feat_type, channel=0, log=False, delta=0, cmvn=False):
        assert feat_type in ["wav", "complx", "linear", "phase", "mel", "mfcc"]
        assert type(channel) is int
        assert type(log) is bool
        assert type(delta) is int and delta >= 0
        assert type(cmvn) is bool
        return {
            "feat_type": feat_type,
            "channel": channel,
            "log": log,
            "delta": delta,
            "cmvn": cmvn,
        }

    def forward(self, wavs=None, feat_list=None, wavs_len=None):
        # wavs: (*, channel_size, max_len)
        # feat_list, mam_list: [{feat_type: 'mfcc', channel: 0, log: False, delta: 2, cmvn: 'True'}, ...]
        # wavs_len: [len1, len2, ...]

        feat_list = self._check_list(feat_list)
        if wavs is None:
            max_channel_id = max(
                [int(args["channel"]) if "channel" in args else 0 for args in feat_list]
            )
            wavs = self._pseudo_wavs[0].view(1, 1, -1).repeat(1, max_channel_id + 1, 1)
        assert wavs.dim() >= 3

        # find length of wavs from padded tensor
        if wavs_len is None:
            wavs_len = []
            for wav in wavs:
                nonzero_index = wav.nonzero()
                if len(nonzero_index) == 0:
                    wavs_len.append(wav.size(-1))  # when all elements are zero
                else:
                    wavs_len.append(nonzero_index[:, -1].max().item() + 1)
            wavs = pad_sequence(
                [
                    wav[:wav_len].transpose(-1, -2)
                    for wav, wav_len in zip(wavs, wavs_len)
                ],
                batch_first=True,
            ).transpose(-1, -2)

        wav = wavs.unsqueeze(2)
        shape = wavs.size()
        complx = self._stft(wavs.reshape(-1, shape[-1]), window=self._window)
        complx = complx.reshape(shape[:-1] + complx.shape[-3:])
        # complx: (*, channel_size, feat_dim, max_len, 2)
        linear, phase = self._magphase(complx)
        mel = self._melscale(linear)
        mfcc = self._mfcc_trans(wavs)
        complx = complx.transpose(-1, -2).reshape(*mfcc.shape[:2], -1, mfcc.size(-1))
        # complx, linear, phase, mel, mfcc: (*, channel_size, feat_dim, max_len)

        def select_feat(
            variables, feat_type, channel=0, log=False, delta=0, cmvn=False
        ):
            raw_feat = variables[feat_type].select(dim=-3, index=channel)
            # apply log scale
            if bool(log):
                raw_feat = (raw_feat + self.eps).log()
            feats = [raw_feat.contiguous()]
            # apply delta for features
            for _ in range(int(delta)):
                feats.append(compute_deltas(feats[-1]))
            feats = torch.cat(feats, dim=-2)
            downsample_rate = wavs.size(-1) / feats.size(-1)
            feats_len = [round(length / downsample_rate) for length in wavs_len]
            # apply cmvn
            if bool(cmvn):
                cmvn_feats = []
                for feat, feat_len in zip(feats, feats_len):
                    feat = feat[:, :feat_len]
                    cmvn_feat = (feat - feat.mean(dim=-1, keepdim=True)) / (
                        feat.std(dim=-1, keepdim=True) + self.eps
                    )
                    cmvn_feats.append(cmvn_feat.transpose(-1, -2))
                feats = pad_sequence(cmvn_feats, batch_first=True).transpose(-1, -2)
            return feats
            # return: (*, feat_dim, max_len)

        local_variables = locals()
        return self._transpose_list(
            [select_feat(local_variables, **args) for args in feat_list]
        )
        # return: [(*, max_len, feat_dim), ...]

    def istft(self, linears=None, phases=None, linear_power=2, complxs=None):
        assert complxs is not None or (linears is not None and phases is not None)
        # complxs: (*, n_freq, max_feat_len, 2) or (*, max_feat_len, n_freq * 2)
        # linears, phases: (*, max_feat_len, n_freq)

        if complxs is None:
            linears, phases = self._transpose_list([linears, phases])
            complxs = linears.pow(1 / linear_power).unsqueeze(-1) * torch.stack(
                [phases.cos(), phases.sin()], dim=-1
            )
        if complxs.size(-1) != 2:
            # treat complxs as: (*, max_feat_len, n_freq * 2)
            shape = complxs.size()
            complxs = complxs.view(*shape[:-1], -1, 2).transpose(-2, -3).contiguous()
        # complxs: (*, n_freq, max_feat_len, 2)

        return self._istft(complxs, window=self._window)
        # return: (*, max_wav_len)

    def test_istft(self, wavs=None, atol=1e-6):
        # wavs: (*, channel_size, max_wav_len)
        channel1, channel2 = 0, 1
        max_channel_id = max(channel1, channel2)

        if wavs is None:
            wavs = self._pseudo_wavs[: max_channel_id + 1].unsqueeze(0)
        assert wavs.size(-2) > max_channel_id

        feat_list = [
            {"feat_type": "complx", "channel": channel1},
            {"feat_type": "linear", "channel": channel2},
            {"feat_type": "phase", "channel": channel2},
        ]
        complxs, linears, phases = self.forward(wavs, feat_list)
        assert torch.allclose(
            wavs.select(dim=-2, index=channel1), self.istft(complxs=complxs), atol=atol
        )
        assert torch.allclose(
            wavs.select(dim=-2, index=channel2),
            self.istft(linears=linears, phases=phases),
            atol=atol,
        )
        print("[Preprocessor] test passed: stft -> istft")
