import os
import torch

from utility.download import _gdriveids_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def apc(ckpt, config, *args, **kwargs):
    f"""
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(config)
    upstream = _UpstreamExpert(ckpt, config)
    return upstream


def apc_gdriveid(ckpt, config, refresh=False, *args, **kwargs):
    f"""
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            config (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return apc(*_gdriveids_to_filepaths(ckpt, config, refresh=refresh))


def apc_default(refresh=False, *args, **kwargs):
    f"""
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    ckpt = '17EUmcitnDCZ1vBTDR7Qq_JP3j-z1cO74'
    config = '1N7oAecCBAEqqSS4QPXs_gN2RkSfd5i5-'
    return apc_gdriveid(ckpt, config, refresh=refresh)
