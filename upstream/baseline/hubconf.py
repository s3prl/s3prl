import os

from .expert import UpstreamExpert as _UpstreamExpert


def baseline_local(config, *args, **kwargs):
    """
        Baseline feature
            config: PATH
    """
    assert os.path.isfile(config)
    return _UpstreamExpert(config, *args, **kwargs)


def baseline(*args, **kwargs):
    """
        Baseline feature - Fbank, or Mel-scale spectrogram
    """
    return fbank(*args, **kwargs)


def spectrogram(*args, **kwargs):
    """
        Baseline feature - Linear-scale spectrogram
    """
    kwargs['config'] = os.path.join(os.path.dirname(__file__), 'spectrogram.yaml')
    return baseline_local(*args, **kwargs)


def fbank(*args, **kwargs):
    """
        Baseline feature - Fbank, or Mel-scale spectrogram
    """
    kwargs['config'] = os.path.join(os.path.dirname(__file__), 'fbank.yaml')
    return baseline_local(*args, **kwargs)


def mfcc(*args, **kwargs):
    """
        Baseline feature - MFCC
    """
    kwargs['config'] = os.path.join(os.path.dirname(__file__), 'mfcc.yaml')
    return baseline_local(*args, **kwargs)


def mel(*args, **kwargs):
    """
        Baseline feature - Mel
    """
    kwargs['config'] = os.path.join(os.path.dirname(__file__), 'mel.yaml')
    return baseline_local(*args, **kwargs)


def linear(*args, **kwargs):
    """
        Baseline feature - Linear
    """
    kwargs['config'] = os.path.join(os.path.dirname(__file__), 'linear.yaml')
    return baseline_local(*args, **kwargs)