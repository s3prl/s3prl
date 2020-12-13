from .expert import UpstreamExpert as _UpstreamExpert


def example(ckpt=None, config=None, refresh=False, *args, **kwargs):
    """
        A minimal example for registering an upstream
            ckpt, config, refresh: Please check benchmark/runner.py:35
    """
    return _UpstreamExpert()
