from .expert import UpstreamExpert as _UpstreamExpert


def hf_wav2vec2_custom(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)
