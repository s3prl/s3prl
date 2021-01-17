import os
import torch

from utility.download import _gdriveids_to_filepaths
from ..tera.hubconf import tera_logMelBase_time_AdamW_b32_1m_960hr as mockingjay_logMelBase_AdamW_b32_1m_960hr


def mockingjay_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def mockingjay_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def mockingjay(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_logMelBase_AdamW_b32_1m_960hr(refresh=refresh, *args, **kwargs)
