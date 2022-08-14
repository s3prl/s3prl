from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def decoar_layers_custom(ckpt: str, refresh=False, *args, **kwargs):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    return _UpstreamExpert(ckpt, *args, **kwargs)


def decoar_layers_local(*args, **kwargs):
    return decoar_layers_custom(*args, **kwargs)


def decoar_layers_url(*args, **kwargs):
    return decoar_layers_custom(*args, **kwargs)


def decoar_layers(*args, refresh=False, **kwargs):
    """
    The apc standard model on 360hr
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/checkpoint_decoar.pt"
    return decoar_layers_url(*args, refresh=refresh, **kwargs)
