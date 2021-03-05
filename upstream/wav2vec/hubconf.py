# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec/hubconf.py ]
#   Synopsis     [ the wav2vec torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
#-------------#
from utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec_local(ckpt, feature_selection=None, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (str): 'c' (default) or 'z'
    """
    assert os.path.isfile(ckpt)
    if feature_selection not in ['c', 'z']:
        feature_selection = 'c'
    return _UpstreamExpert(ckpt, feature_selection)


def wav2vec_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): URL
            feature_selection (str): 'c' or 'z'
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def wav2vec(refresh=False, *args, **kwargs):
    """
        The default model - Large model
            feature_selection (str): 'c' or 'z'
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec_large(refresh=refresh, *args, **kwargs)


def wav2vec_large(refresh=False, *args, **kwargs):
    """
        The Large model
            feature_selection (str): 'c' or 'z'
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt'
    return wav2vec_url(refresh=refresh, *args, **kwargs)
