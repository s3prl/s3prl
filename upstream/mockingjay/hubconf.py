import os
import torch

from utility.download import _gdriveids_to_filepaths
from upstream.mockingjay.expert import UpstreamExpert as _UpstreamExpert


def mockingjay_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def mockingjay_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def mockingjay(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = '1-JwGlb3JXXnnXqKL9WVtbv34RR0o6BAX'
    return mockingjay_gdriveid(refresh=refresh, *args, **kwargs)


def mockingjay_logMelBase_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled speech: 100hr
        Usage note: this is identical to `-u tera_logMelBase_time_AdamW_b32_200k_100hr`
    """
    kwargs['ckpt'] = '1-JwGlb3JXXnnXqKL9WVtbv34RR0o6BAX'
    return mockingjay_gdriveid(refresh=refresh, *args, **kwargs)