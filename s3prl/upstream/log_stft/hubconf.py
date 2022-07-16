import os

from .expert import UpstreamExpert as _UpstreamExpert


def stft_mag(model_config, *args, **kwargs):
    assert os.path.isfile(model_config)
    return _UpstreamExpert(model_config, *args, **kwargs)
