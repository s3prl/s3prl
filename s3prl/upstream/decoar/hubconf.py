from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def decoar_custom(ckpt: str, refresh=False, *args, **kwargs):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    return _UpstreamExpert(ckpt, *args, **kwargs)


def decoar_local(*args, **kwargs):
    return decoar_custom(*args, **kwargs)


def decoar_url(*args, **kwargs):
    return decoar_custom(*args, **kwargs)


def decoar(refresh=False, *args, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/checkpoint_decoar.pt"
    return decoar_url(*args, refresh=refresh, **kwargs)
