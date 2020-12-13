from .expert import UpstreamExpert as _UpstreamExpert


def example(*args, **kwargs):
    return _UpstreamExpert()
