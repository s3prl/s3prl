# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec/hubconf.py ]
#   Synopsis     [ the wav2vec torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import LegacyUpstreamExpert as _LegacyUpstreamExpert
from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec_custom(
    ckpt: str, *args, legacy: bool = False, refresh: bool = False, **kwargs
):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)


def wav2vec_custom(
    ckpt: str, *args, legacy: bool = False, refresh: bool = False, **kwargs
):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    assert os.path.isfile(ckpt)
    if legacy:
        return _LegacyUpstreamExpert(ckpt, *args, **kwargs)
    else:
        return _UpstreamExpert(ckpt, *args, **kwargs)


def wav2vec_local(*args, **kwargs):
    return wav2vec_custom(*args, **kwargs)


def wav2vec_url(*args, **kwargs):
    return wav2vec_custom(*args, **kwargs)


def wav2vec(refresh=False, *args, **kwargs):
    """
    The default model - Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec_large(refresh=refresh, *args, **kwargs)


def wav2vec_large(refresh=False, legacy=False, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec_large.pt"
    return wav2vec_custom(refresh=refresh, legacy=legacy, **kwargs)
