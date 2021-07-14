# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/spec_augment/hubconf.py ]
#   Synopsis     [ the spec augment torch hubconf ]
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


def spec_augment_local(ckpt, feature_selection=None, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (int): -1 (default, the last layer) or an int in range(0, max_layer_num)
    """
    assert os.path.isfile(ckpt)
    if feature_selection is None:
        feature_selection = -1
    return _UpstreamExpert(ckpt, feature_selection, *args, **kwargs)


def spec_augment_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return spec_augment_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def spec_augment_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return spec_augment_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def spec_augment(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/spz3yulaye8ppgr/states-100000.ckpt?dl=0'
    return spec_augment_url(refresh=refresh, *args, **kwargs)