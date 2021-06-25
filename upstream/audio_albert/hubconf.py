# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/audio_albert/hubconf.py ]
#   Synopsis     [ the audio albert torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from utility.download import _gdriveids_to_filepaths, _urls_to_filepaths
from upstream.audio_albert.expert import UpstreamExpert as _UpstreamExpert


def audio_albert_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (int): -1 (default, the last layer) or an int in range(0, max_layer_num)
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def audio_albert_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return audio_albert_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def audio_albert_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return audio_albert_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def audio_albert(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return audio_albert_960hr(refresh=refresh, *args, **kwargs)


###########
# ALIASES #
###########


def audio_albert_960hr(refresh=False, *args, **kwargs):
    """
        The audio albert base model on 960hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    return audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1(refresh=refresh, *args, **kwargs)


##########
# 960 HR #
##########


def audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1(refresh=False, *args, **kwargs):
    """
        Feature: 80-dim log Mel
        Alteration: time
        Optimizer: AdamW
        Batch size: 32
        Total steps: 1M
        Unlabled Speech: 960hr
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/3wgynxmod77ha1z/states-1000000.ckpt?dl=0'
    return audio_albert_url(refresh=refresh, *args, **kwargs)