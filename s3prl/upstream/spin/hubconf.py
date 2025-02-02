import logging

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)


def spin_custom(
    ckpt: str,
    refresh: bool = False,
    **kwargs,
):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    return _UpstreamExpert(ckpt, **kwargs)


def spin_local(*args, **kwargs):
    return spin_custom(*args, **kwargs)


def spin_url(*args, **kwargs):
    return spin_custom(*args, **kwargs)


def spin_hubert_128(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_128.ckpt"
    return spin_custom(refresh=refresh, **kwargs)


def spin_hubert_256(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_256.ckpt"
    return spin_custom(refresh=refresh, **kwargs)


def spin_hubert_512(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_512.ckpt"
    return spin_custom(refresh=refresh, **kwargs)


def spin_hubert_1024(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_1024.ckpt"
    return spin_custom(refresh=refresh, **kwargs)


def spin_hubert_2048(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_2048.ckpt"
    return spin_custom(refresh=refresh, **kwargs)


def spin_wavlm_128(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_128.ckpt"
    return spin_custom(refresh=refresh, **kwargs)


def spin_wavlm_256(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_256.ckpt"
    return spin_custom(refresh=refresh, **kwargs)


def spin_wavlm_512(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_512.ckpt"
    return spin_custom(refresh=refresh, **kwargs)


def spin_wavlm_1024(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_1024.ckpt"
    return spin_custom(refresh=refresh, **kwargs)


def spin_wavlm_2048(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_2048.ckpt"
    return spin_custom(refresh=refresh, **kwargs)
