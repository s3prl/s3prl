from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec2_hf(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)
