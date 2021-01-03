import os
from .expert import UpstreamExpert as _UpstreamExpert

def asr_feat(config, *args, **kwargs):
    """
        The feature dedicated for ASR, only for debugging
            config: PATH
    """
    assert os.path.isfile(config)
    return _UpstreamExpert(config, *args, **kwargs)
