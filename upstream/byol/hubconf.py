# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol/hubconf.py ]
#   Synopsis     [ the byol torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
#-------------#
from utility.download import _gdriveids_to_filepaths, _urls_to_filepaths
from upstream.byol.expert import UpstreamExpert as _UpstreamExpert


def byol_local(ckpt, feature_selection=None, model_config=None, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (int): -1 (default, the last layer) or an int in range(0, max_layer_num)
    """
    assert os.path.isfile(ckpt)
    if feature_selection is None:
        feature_selection = -1
    if model_config is not None:
        assert os.path.isfile(model_config)
    return _UpstreamExpert(ckpt, feature_selection, model_config, *args, **kwargs)


def byol_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return byol_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def byol_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return byol_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def byol(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'todo'
    return byol_url(refresh=refresh, *args, **kwargs)