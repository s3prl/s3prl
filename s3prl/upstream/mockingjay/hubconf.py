# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/hubconf.py ]
#   Synopsis     [ the mockingjay torch hubconf ]
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


def mockingjay_local(ckpt, options_config=None, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (int): -1 (default, the last layer) or an int in range(0, max_layer_num)
    """
    assert os.path.isfile(ckpt)
    if options_config is not None:
        assert os.path.isfile(options_config)
    return _UpstreamExpert(ckpt, options_config, *args, **kwargs)


def mockingjay_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def mockingjay_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return mockingjay_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def mockingjay(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_origin(refresh=refresh, *args, **kwargs)


###########
# ALIASES #
###########


def mockingjay_origin(refresh=False, *args, **kwargs):
    """
        The mockingjay large model on 360hr, with Lel as input and Linear as target
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1(refresh=refresh, *args, **kwargs)


def mockingjay_100hr(refresh=False, *args, **kwargs):
    """
        The mockingjay base model on 100hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_logMelBase_T_AdamW_b32_200k_100hr(refresh=refresh, *args, **kwargs)


def mockingjay_960hr(refresh=False, *args, **kwargs):
    """
        The mockingjay base model on 960hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1(refresh=refresh, *args, **kwargs)


##########
# 100 HR #
##########


def mockingjay_logMelBase_T_AdamW_b32_200k_100hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time
        Optimizer: AdamW
        Batch size: 32
        Total steps: 200k
        Unlabled Speech: 100hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/luorglf8mdg67l2/states-200000.ckpt?dl=0'
    return mockingjay_url(refresh=refresh, *args, **kwargs)


##########
# 360 HR #
##########


def mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel (input) / 201-dim Linear (target)
        Alteration: time
        Optimizer: AdamW
        Batch size: 32
        Total steps: 500k
        Unlabled Speech: 360hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/zwsfa6w2iy2cc68/states-500000.ckpt?dl=0'
    return mockingjay_url(refresh=refresh, *args, **kwargs)


##########
# 960 HR #
##########


def mockingjay_logMelBase_T_AdamW_b32_1m_960hr(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/jzx0xggk663jev6/states-1000000.ckpt?dl=0'
    return mockingjay_url(refresh=refresh, *args, **kwargs)


def mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
        Differences: Dropout of 0.1 (instead of 0.3)
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/7f9z6dzc7oix6qv/states-1000000.ckpt?dl=0'
    return mockingjay_url(refresh=refresh, *args, **kwargs)


def mockingjay_logMelBase_T_AdamW_b32_1m_960hr_seq3k(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
        Differences: sequence length of 3k (instead of 1.5k)
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/qnnvdrai2tfmjmh/states-1000000.ckpt?dl=0'
    return mockingjay_url(refresh=refresh, *args, **kwargs)