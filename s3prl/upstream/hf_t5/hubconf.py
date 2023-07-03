from .expert import UpstreamExpert as _UpstreamExpert

def hf_t5_custom(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)