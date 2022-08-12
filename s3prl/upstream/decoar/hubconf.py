###############
# IMPORTATION #
###############
import os
#-------------#
from s3prl.util.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def decoar_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (str): 'c' (default) or 'z'
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def decoar_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return decoar_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)

def decoar(refresh=False, *args, **kwargs):
    """
        The apc standard model on 360hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/decoar/resolve/main/checkpoint_decoar.pt"
    return decoar_url(refresh=refresh, *args, **kwargs)
