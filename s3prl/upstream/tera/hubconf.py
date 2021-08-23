# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/tera/hubconf.py ]
#   Synopsis     [ the tera torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
#-------------#
from s3prl.utility.download import _gdriveids_to_filepaths, _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def tera_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (int): -1 (default, the last layer) or an int in range(0, max_layer_num)
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
    return tera_960hr(refresh, *args, **kwargs)


###########
# ALIASES #
###########


def tera_100hr(refresh=False, *args, **kwargs):
    """
        The tera base model on 100hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    return tera_logMelBase_T_F_M_AdamW_b32_200k_100hr(refresh, *args, **kwargs)


def tera_960hr(refresh=False, *args, **kwargs):
    """
        The tera base model on 960hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    return tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1(refresh, *args, **kwargs)


##########
# 100 HR #
##########


def tera_logMelBase_T_F_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
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


def tera_logMelBase_T_F_M_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
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


def tera_logMelBase_T_F_AdamW_b32_1m_960hr(refresh=False, *args, **kwargs):
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


def tera_logMelBase_T_F_AdamW_b32_1m_960hr_drop1(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
        Differences: Dropout of 0.1 (instead of 0.3)
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/2ekbt2gxlkbvfz0/states-1000000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_T_F_AdamW_b32_1m_960hr_seq3k(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
        Differences: sequence length of 3k (instead of 1.5k)
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/tfysinbalpm3gsj/states-1000000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


def tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time + freq + mag
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
        Differences: Dropout of 0.1 (instead of 0.3)
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/xdoj9wdo87lztv1/states-1000000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)


#####################
# Other Feat 100 HR #
#####################


def tera_fbankBase_T_F_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 240-dim fbank
        Alteration: time + freq
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/i32ob29m6afufot/states-200000.ckpt?dl=0'
    return tera_url(refresh=refresh, *args, **kwargs)