# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/hubconf.py ]
#   Synopsis     [ the HuBERT torch hubconf ]
#   Author       [ S3PRL / Kushal Lakhotia ]
"""*********************************************************************************************"""


import logging
import os
import time
from pathlib import Path

from filelock import FileLock

from s3prl.util.download import _urls_to_filepaths

from .convert import load_and_convert_fairseq_ckpt
from .expert import LegacyUpstreamExpert as _LegacyUpstreamExpert
from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)

NEW_ENOUGH_SECS = 2.0


def hubert_custom(
    ckpt: str,
    legacy: bool = False,
    fairseq: bool = False,
    refresh: bool = False,
    **kwargs,
):
    assert not (legacy and fairseq), (
        "The option 'legacy' will directly load a fairseq checkpoint, "
        "while the option 'fairseq' will first convert the fairseq checkpoint to "
        "be fairseq indenpendent and then load the checkpoint. "
        "These two options cannot be used jointly."
    )

    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    if fairseq:
        ckpt: Path = Path(ckpt)
        converted_ckpt = ckpt.parent / f"{ckpt.stem}.converted.pt"
        lock_file = Path(str(converted_ckpt) + ".lock")

        logger.info(f"Converting a fairseq checkpoint: {ckpt}")
        logger.info(f"To: {converted_ckpt}")

        with FileLock(str(lock_file)):
            if not converted_ckpt.is_file() or (
                refresh and (time.time() - os.path.getmtime(ckpt)) > NEW_ENOUGH_SECS
            ):
                load_and_convert_fairseq_ckpt(ckpt, converted_ckpt)

        ckpt = converted_ckpt

    assert os.path.isfile(ckpt)
    if legacy:
        return _LegacyUpstreamExpert(ckpt, **kwargs)
    else:
        return _UpstreamExpert(ckpt, **kwargs)


def hubert_local(*args, **kwargs):
    return hubert_custom(fairseq=True, *args, **kwargs)


def hubert_url(*args, **kwargs):
    return hubert_custom(fairseq=True, *args, **kwargs)


def hubert(refresh=False, *args, **kwargs):
    """
    The default model - Base
        refresh (bool): whether to download ckpt/config again if existed
    """
    return hubert_base(refresh=refresh, *args, **kwargs)


def hubert_base(refresh=False, legacy=False, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_base_ls960.pt"
    return hubert_custom(refresh=refresh, legacy=legacy, **kwargs)


def hubert_large_ll60k(refresh=False, legacy=False, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_large_ll60k.pt"
    return hubert_custom(refresh=refresh, legacy=legacy, **kwargs)


def hubert_base_robust_mgr(refresh=False, legacy=False, **kwargs):
    """
    The Base model, continually trained with Libri 960 hr with Musan noise, Gaussian noise and Reverberation.
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/kphuang68/HuBERT_base_robust_mgr/resolve/main/HuBERT_base_robust_mgr_best_loss_2.7821.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/HuBERT_base_robust_mgr_best_loss_2.7821.pt"
    return hubert_custom(refresh=refresh, legacy=legacy, **kwargs)


def mhubert_base_vp_en_es_fr_it3(refresh=False, **kwds):
    kwds[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/mhubert_base_vp_en_es_fr_it3.pt"
    return hubert_custom(refresh=refresh, **kwds)


def contentvec(refresh=False, **kwds):
    kwds[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/contentvec_km100.pt"
    return hubert_custom(refresh=refresh, **kwds)


def contentvec_km100(refresh=False, **kwds):
    kwds[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/contentvec_km100.pt"
    return hubert_custom(refresh=refresh, **kwds)


def contentvec_km500(refresh=False, **kwds):
    kwds[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/contentvec_km500.pt"
    return hubert_custom(refresh=refresh, **kwds)
