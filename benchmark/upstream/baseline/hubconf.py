import os

from .expert import UpstreamExpert as _UpstreamExpert


def baseline(config, *args, **kwargs):
    assert os.path.isfile(config)
    return _UpstreamExpert(config)


def baseline_default(*args, **kwargs):
    """
        Default baseline feature - Fbank, or Mel-scale spectrogram
    """
    return _UpstreamExpert('benchmark/upstream/baseline/fbank.yaml')


def spectrogram(*args, **kwargs):
    """
        Baseline feature - Linear-scale spectrogram
    """
    return _UpstreamExpert('benchmark/upstream/baseline/spectrogram.yaml')


def fbank(*args, **kwargs):
    """
        Baseline feature - Fbank, or Mel-scale spectrogram
    """
    return _UpstreamExpert('benchmark/upstream/baseline/fbank.yaml')


def mfcc(*args, **kwargs):
    """
        Baseline feature - MFCC
    """
    return _UpstreamExpert('benchmark/upstream/baseline/mfcc.yaml')
