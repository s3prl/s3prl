import os
import torch
#-------------#
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def decoar2_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (str): 'c' (default) or 'z'
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def decoar2_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return decoar2_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)

def decoar2(refresh=False, *args, **kwargs):
    """
        The apc standard model on 360hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://speech-representation.s3.us-west-2.amazonaws.com/checkpoint_decoar2.pt'
    return decoar2_url(refresh=refresh, *args, **kwargs)
