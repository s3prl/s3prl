# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/decoar/hubconf.py ]
#   Synopsis     [ the decoar torch hubconf ]
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


def decoar_local(ckpt, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, **kwargs)


def decoar_url(ckpt, refresh=False, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    ckpt = _urls_to_filepaths(ckpt, refresh=refresh)
    return decoar_local(ckpt, **kwargs)


def decoar(refresh=False, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/0x43bfv8xcmccr3/decoar-encoder-29b8e2ac.params?dl=0'
    return decoar_url(refresh=refresh, **kwargs)
