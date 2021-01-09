import os
import torch

from utility.download import _gdriveids_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def phone_1hidden_posterior(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def phone_1hidden_posterior_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return phone_1hidden_posterior(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def phone_1hidden_posterior_tera(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = '1xu5rzW2xPKHEjvdhIZAXKLuEbanp8jvn'
    return phone_1hidden_posterior_gdriveid(refresh=refresh, *args, **kwargs)
