import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def lighthubert_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def lighthubert_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return lighthubert_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def lighthubert(refresh=False, *args, **kargs):
    """
    The default model - Small
    refresh (bool): whether to download ckpt/config again if existed
    """
    return lighthubert_small(refresh=refresh, *args, **kargs)


def lighthubert_small(refresh=False, *args, **kwargs):
    """
    The small model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/mechanicalsea/lighthubert/resolve/main/lighthubert_small.pt"
    return lighthubert_url(refresh=refresh, *args, **kwargs)


def lighthubert_base(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/mechanicalsea/lighthubert/resolve/main/lighthubert_base.pt"
    return lighthubert_url(refresh=refresh, *args, **kwargs)


def lighthubert_stage1(refresh=False, *args, **kwargs):
    """
    The Stage1 model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/mechanicalsea/lighthubert/resolve/main/lighthubert_stage1.pt"
    return lighthubert_url(refresh=refresh, *args, **kwargs)
