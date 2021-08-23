# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/cpc/hubconf.py ]
#   Synopsis     [ the cpc torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def cpc_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def cpc_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return cpc_local(_urls_to_filepaths(ckpt), *args, **kwargs)


def modified_cpc(refresh=False, *args, **kwargs):
    """
        The model from official repository
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt'
    return cpc_url(refresh=refresh, *args, **kwargs)
