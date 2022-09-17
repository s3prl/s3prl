# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/apc/audio.py ]
#   Synopsis     [ audio utility functions for apc ]
#   Author       [ S3PRL / Tzu-Wei Sung (https://github.com/WindQAQ) ]
#   Reference    [ Utterence-wised zero mean unit variance from:
#                  https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch/blob/master/src/audio.py ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
import torchaudio
from torchaudio.compliance import kaldi

############
# CONSTANT #
############
WINDOW_TYPE = "hanning"
SAMPLE_RATE = 16000


class CMVN(torch.jit.ScriptModule):
    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=2, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
        # so perform normalization on dim=2 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization."
            )

        self.mode = mode
        self.dim = dim
        self.eps = eps

    @torch.jit.script_method
    def forward(self, x):
        if self.mode == "global":
            return (x - x.mean(self.dim, keepdim=True)) / (
                self.eps + x.std(self.dim, keepdim=True)
            )

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)


class FeatureExtractor(nn.Module):
    """Feature extractor, transforming file path to Mel spectrogram"""

    def __init__(
        self,
        mode="fbank",
        num_mel_bins=128,
        decode_wav=False,
        apply_cmvn=False,
        target_length=100,
        **kwargs
    ):
        super(FeatureExtractor, self).__init__()
        # ToDo: Other surface representation
        assert mode == "fbank", "Only Mel-spectrogram implemented"
        self.mode = mode
        self.target_length = target_length
        self.extract_fn = kaldi.fbank
        self.apply_cmvn = apply_cmvn
        if self.apply_cmvn:
            self.cmvn = CMVN()
        self.num_mel_bins = num_mel_bins
        self.kwargs = kwargs
        self.decode_wav = decode_wav
        if self.decode_wav:
            # HACK: sox cannot deal with wav with incorrect file length
            torchaudio.set_audio_backend("soundfile")

    def _load_file(self, filepath):
        if self.decode_wav:
            waveform, sample_rate = torchaudio.load_wav(filepath)
        else:
            waveform, sample_rate = torchaudio.load(filepath)
        return waveform, sample_rate

    def forward(self, waveform):
        y = self.extract_fn(
            waveform,
            htk_compat=True,
            sample_frequency=SAMPLE_RATE,
            use_energy=False,
            window_type=WINDOW_TYPE,
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=10,
            **self.kwargs
        )

        y = (y + 4.2677393) / (4.5689974 * 2)
        p = self.target_length - y.shape[0]

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            y = m(y)
        elif p < 0:
            y = y[0 : self.target_length, :]

        # #CMVN
        if self.apply_cmvn:
            y = y.transpose(0, 1).unsqueeze(0)  # TxD -> 1xDxT
            y = self.cmvn(y)
            y = y.squeeze(0).transpose(0, 1)  # 1xDxT -> TxD
        return y

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)

    def create_msg(self):
        """List msg for verbose function"""
        msg = "Audio spec.| Audio feat. = {}\t\t| feat. dim = {}\t| CMVN = {}".format(
            self.mode, self.num_mel_bins, self.apply_cmvn
        )
        return [msg]


def create_transform(audio_config):
    feat_type = audio_config.pop("feat_type")
    feat_dim = audio_config.pop("feat_dim")
    decode_wav = audio_config.pop("decode_wav", False)
    apply_cmvn = False
    transforms = FeatureExtractor(
        feat_type, feat_dim, decode_wav, apply_cmvn, **audio_config
    )
    return transforms, feat_dim
