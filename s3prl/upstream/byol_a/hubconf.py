# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol_a/hubconf.py ]
#   Synopsis     [ the BYOL-A torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

from pathlib import Path as _Path

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

DEFAULT_CONFIG_PATH = _Path(__file__).parent / "config.yaml"


def _byol_a_2048(refresh=False, **kwds):
    ckpt = _urls_to_filepaths(
        "https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth",
        refresh=refresh,
    )
    del kwds['ckpt'], kwds['model_config']  # to prevent duplicates in the kwds.
    return _UpstreamExpert(ckpt, DEFAULT_CONFIG_PATH, 2048, **kwds)


def _byol_a_1024(refresh=False, **kwds):
    ckpt = _urls_to_filepaths(
        "https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d1024.pth",
        refresh=refresh,
    )
    del kwds['ckpt'], kwds['model_config']
    return _UpstreamExpert(ckpt, DEFAULT_CONFIG_PATH, 1024, **kwds)


def _byol_a_512(refresh=False, **kwds):
    ckpt = _urls_to_filepaths(
        "https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d512.pth",
        refresh=refresh,
    )
    del kwds['ckpt'], kwds['model_config']
    return _UpstreamExpert(ckpt, DEFAULT_CONFIG_PATH, 512, **kwds)


### Add entries for each downstream tasks


def byol_a_2048_calcnorm(refresh=False, **kwds):
    """Calculate downstream task statistics using this model."""
    return _byol_a_2048(**kwds)


def byol_a_1024_calcnorm(refresh=False, **kwds):
    """Calculate downstream task statistics using this model."""
    return _byol_a_1024(**kwds)


def byol_a_512_calcnorm(refresh=False, **kwds):
    """Calculate downstream task statistics using this model."""
    return _byol_a_512(**kwds)


def byol_a_2048_LS(refresh=False, **kwds):
    """BYOL-A d=2048 for LibriSpeech tasks."""
    return _byol_a_2048(norm_mean=-8.5402, norm_std=4.5456, **kwds)


def byol_a_1024_LS(refresh=False, **kwds):
    """BYOL-A d=2048 for LibriSpeech tasks."""
    return _byol_a_1024(norm_mean=-8.5402, norm_std=4.5456, **kwds)


def byol_a_512_LS(refresh=False, **kwds):
    """BYOL-A d=2048 for LibriSpeech tasks."""
    return _byol_a_512(norm_mean=-8.5402, norm_std=4.5456, **kwds)


def byol_a_2048_vc1(refresh=False, **kwds):
    """BYOL-A d=2048 for voxceleb1."""
    return _byol_a_2048(norm_mean=-8.9072303, norm_std=4.8924856, **kwds)
