# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/vq_wav2vec/hubconf.py ]
#   Synopsis     [ the vq wav2vec torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import LegacyUpstreamExpert as _LegacyUpstreamExpert
from .expert import UpstreamExpert as _UpstreamExpert


def vq_wav2vec_custom(
    ckpt: str, *args, legacy: bool = False, refresh: bool = False, **kwargs
):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    assert os.path.isfile(ckpt)
    if legacy:
        return _LegacyUpstreamExpert(ckpt, *args, **kwargs)
    else:
        return _UpstreamExpert(ckpt, *args, **kwargs)


def wav2vec2_local(*args, **kwargs):
    return vq_wav2vec_custom(*args, **kwargs)


def wav2vec2_url(*args, **kwargs):
    return vq_wav2vec_custom(*args, **kwargs)


def vq_wav2vec(refresh=False, *args, **kwargs):
    """
    The default model - Large model with context vector
        refresh (bool): whether to download ckpt/config again if existed
    """
    return vq_wav2vec_gumbel(refresh=refresh, *args, **kwargs)


def vq_wav2vec_gumbel(refresh=False, legacy=False, **kwargs):
    """
    The Gumbel model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/vq-wav2vec.pt"
    return vq_wav2vec_custom(refresh=refresh, legacy=legacy, **kwargs)


def vq_wav2vec_kmeans(refresh=False, legacy=False, **kwargs):
    """
    The K-means model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/vq-wav2vec_kmeans.pt"
    return vq_wav2vec_custom(refresh=refresh, legacy=legacy, **kwargs)
