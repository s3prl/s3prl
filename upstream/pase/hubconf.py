# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/pase/hubconf.py ]
#   Synopsis     [ the pase torch hubconf ]
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
from .expert import UpstreamExpert as _UpstreamExpert


def pase_local(ckpt, model_config, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            model_config (str): PATH
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(model_config)
    return _UpstreamExpert(ckpt, model_config, **kwargs)


def pase_url(ckpt, model_config, refresh=False, **kwargs):
    """
        The model from URL
            ckpt (str): URL
            model_config (str): URL
    """
    ckpt = _urls_to_filepaths(ckpt, refresh=refresh)
    model_config = _urls_to_filepaths(model_config, refresh=refresh)
    return pase_local(ckpt, model_config, **kwargs)


def pase_plus(refresh=False, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/p8811o7eadv4pat/FE_e199.ckpt?dl=0'
    kwargs['model_config'] = 'https://www.dropbox.com/s/2p3ouod1k0ekfxn/PASE%2B.cfg?dl=0'
    return pase_url(refresh=refresh, **kwargs)
