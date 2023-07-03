from .expert import UpstreamExpert as _UpstreamExpert

def hf_chinese_hubert_custom(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)