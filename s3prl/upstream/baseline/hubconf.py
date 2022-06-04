# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/baseline/hubconf.py ]
#   Synopsis     [ the baseline torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os

# -------------#
from .expert import UpstreamExpert as _UpstreamExpert


def baseline_local(model_config, *args, **kwargs):
    """
    Baseline feature
        model_config: PATH
    """
    assert os.path.isfile(model_config)
    return _UpstreamExpert(model_config, *args, **kwargs)


def baseline(*args, **kwargs):
    """
    Baseline feature - Fbank, or Mel-scale spectrogram
    """
    return fbank(*args, **kwargs)


def spectrogram(*args, **kwargs):
    """
    Baseline feature - Linear-scale spectrogram
    """
    kwargs["model_config"] = os.path.join(os.path.dirname(__file__), "spectrogram.yaml")
    return baseline_local(*args, **kwargs)


def fbank(*args, **kwargs):
    """
    Baseline feature - Fbank, or Mel-scale spectrogram
    """
    kwargs["model_config"] = os.path.join(os.path.dirname(__file__), "fbank.yaml")
    return baseline_local(*args, **kwargs)


def fbank_no_cmvn(*args, **kwargs):
    """
    Baseline feature - Fbank, or Mel-scale spectrogram
    """
    kwargs["model_config"] = os.path.join(
        os.path.dirname(__file__), "fbank_no_cmvn.yaml"
    )
    return baseline_local(*args, **kwargs)


def mfcc(*args, **kwargs):
    """
    Baseline feature - MFCC
    """
    kwargs["model_config"] = os.path.join(os.path.dirname(__file__), "mfcc.yaml")
    return baseline_local(*args, **kwargs)


def mel(*args, **kwargs):
    """
    Baseline feature - Mel
    """
    kwargs["model_config"] = os.path.join(os.path.dirname(__file__), "mel.yaml")
    return baseline_local(*args, **kwargs)


def linear(*args, **kwargs):
    """
    Baseline feature - Linear
    """
    kwargs["model_config"] = os.path.join(os.path.dirname(__file__), "linear.yaml")
    return baseline_local(*args, **kwargs)
