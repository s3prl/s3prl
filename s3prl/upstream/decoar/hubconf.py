import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def decoar2_custom(ckpt: str, *args, refresh=False, **kwargs):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    return _UpstreamExpert(ckpt, *args, **kwargs)


def decoar_local(ckpt, *args, **kwargs):
    return decoar2_custom(*args, **kwargs)


def decoar_url(*args, **kwargs):
    return decoar2_custom(*args, **kwargs)


def decoar(refresh=False, *args, **kwargs):
    """
    The apc standard model on 360hr
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/decoar/resolve/main/checkpoint_decoar.pt"
    return decoar_url(refresh=refresh, *args, **kwargs)
