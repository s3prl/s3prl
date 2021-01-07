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
    kwargs['ckpt'] = '1g2RBcl4xxvpDtA7eeSIGgygNrMGMUxu2'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = '1-JwGlb3JXXnnXqKL9WVtbv34RR0o6BAX'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_logMelBase_freq_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = '1ZqTQx6vCqSPFbqtDwL-LjKKpoKrga3cz'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_logMelBase_mag_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'todo'
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


def tera_logMelBase_time_mag_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = '1eQrzJ138OXqMTQfNEl_rsT1Y_GTMj0mQ'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_logMelBase_freq_mag_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: freq + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'todo'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_freq_mag_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = '12gUQ6RHyujPp7ZoSsr3t1xrsySsoxVOd'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)