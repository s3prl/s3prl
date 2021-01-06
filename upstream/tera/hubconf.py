import os
import torch

from utility.download import _gdriveids_to_filepaths
from upstream.tera.expert import UpstreamExpert as _UpstreamExpert


def tera(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def tera_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return tera(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def tera_default(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = '1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_freq_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = '1g2RBcl4xxvpDtA7eeSIGgygNrMGMUxu2'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_freq_AdamW_b1024_6k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq
        Optimizer: AdamW
        Batch size: 1024
        Total steps: 6k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = '1OEBYNaHci0A4Pg3hnZ3Ph6O1nhLKK2GY'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_fbankBase_time_freq_AdamW_b32_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim fbank + delta 2
        Alteration: time + freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = '17LjWgkItjBY5XNPu4A5ezMherQ7az4DE'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)
