# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/npc/hubconf.py ]
#   Synopsis     [ the npc torch hubconf ]
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


def npc_local(ckpt, feature_selection, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (str): unmasked-3 or masked
    """
    assert os.path.isfile(ckpt)
    feature_selection = feature_selection or 'unmasked-3'
    return _UpstreamExpert(ckpt, feature_selection, *args, **kwargs)


def npc_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return npc_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def npc_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return npc_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def npc(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return npc_360hr(refresh=refresh, *args, **kwargs)


def npc_360hr(refresh=False, *args, **kwargs):
    """
        The npc standard model on 360hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/o4zpjz6xncbij8p/npc_default.ckpt?dl=0'
    return npc_url(refresh=refresh, *args, **kwargs)


def npc_960hr(refresh=False, *args, **kwargs):
    """
        The npc standard model on 960hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/7ep0v60ym136bpb/npc_960hr.ckpt?dl=0'
    return npc_url(refresh=refresh, *args, **kwargs)