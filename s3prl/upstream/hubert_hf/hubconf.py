from .expert import UpstreamExpert as _UpstreamExpert


def hubert_hf(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)
