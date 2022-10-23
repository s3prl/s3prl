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


def byol_a_2048(refresh=False, **kwds):
    ckpt = _urls_to_filepaths(
        "https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth",
        refresh=refresh,
    )
    return _UpstreamExpert(ckpt, DEFAULT_CONFIG_PATH, 2048, **kwds)


def byol_a_1024(refresh=False, **kwds):
    ckpt = _urls_to_filepaths(
        "https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d1024.pth",
        refresh=refresh,
    )
    return _UpstreamExpert(ckpt, DEFAULT_CONFIG_PATH, 1024, **kwds)


def byol_a_512(refresh=False, **kwds):
    ckpt = _urls_to_filepaths(
        "https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d512.pth",
        refresh=refresh,
    )
    return _UpstreamExpert(ckpt, DEFAULT_CONFIG_PATH, 512, **kwds)
