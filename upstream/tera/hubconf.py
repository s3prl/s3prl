import os
import torch

from utility.download import _gdriveids_to_filepaths, _urls_to_filepaths
from upstream.tera.expert import UpstreamExpert as _UpstreamExpert


def tera_local(ckpt, *args, **kwargs):
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
    return tera_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def tera_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return tera_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def tera(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return tera_logMelBase_time_freq_AdamW_b32_1m_960hr(refresh, *args, **kwargs)


##########
# 100 HR #
##########


def tera_logMelBase_time_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/luorglf8mdg67l2/states-200000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_freq_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/ol873nw97gyjw3l/states-200000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_mag_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/gg9ft4jwtxadh0a/states-200000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_freq_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/o36qt1zgtn3tsep/states-200000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_mag_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/w8g0nqsqpfldij5/states-200000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_freq_mag_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: freq + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/jztwgjsh1o74wj2/states-200000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_freq_mag_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/l9ryl82k64m1lsk/states-200000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


##########
# 960 HR #
##########


def tera_logMelBase_time_AdamW_b32_1m_960hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/jzx0xggk663jev6/states-1000000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_freq_AdamW_b32_1m_960hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/vj047t6257gi1al/states-1000000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_mag_AdamW_b32_1m_960hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/k3cye2cxcsuplrw/states-1000000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_freq_AdamW_b32_1m_960hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/98olxex0m7oy9ta/states-1000000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_mag_AdamW_b32_1m_960hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/i3p2ulhd7robwcl/states-1000000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_freq_mag_AdamW_b32_1m_960hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: freq + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
    """
    kwargs['ckpt'] = ''
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_time_freq_mag_AdamW_b32_1m_960hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/vbagkhl2tgh7fff/states-1000000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


#####################
# Other Feat 100 HR #
#####################


def tera_fbankBase_time_freq_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 240-dim fbank
        Alteration: time + freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/i32ob29m6afufot/states-200000.ckpt?dl=0'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)