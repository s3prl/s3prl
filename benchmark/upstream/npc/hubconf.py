import os
import torch

from utility.download import _gdriveids_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def npc(ckpt, config, *args, **kwargs):
    f"""
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(config)
    upstream = _UpstreamExpert(ckpt, config)
    return upstream


def npc_gdriveid(ckpt, config, refresh=False, *args, **kwargs):
    f"""
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            config (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return npc(*_gdriveids_to_filepaths(ckpt, config, refresh=refresh))


def npc_default(refresh=False, *args, **kwargs):
    f"""
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    ckpt = '1oNmdVEcFMtNt4Rrllxs3ulrEk5TJdioW'
    config = '1ZrfpE0K6us_daQeJTwVENGZn8a4MiPaB'
    return npc_gdriveid(ckpt, config, refresh=refresh)
