from .expert import UpstreamExpert as _UpstreamExpert


def espnet_hubert_custom(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

def espnet_hubert_local(*args, **kwargs):
    return espnet_hubert_custom(*args, **kwargs)


