# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/baseline/hubconf.py ]
#   Synopsis     [ the baseline torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
#-------------#
from .expert import UpstreamExpert as _UpstreamExpert


def stft_mag(model_config, *args, **kwargs):
    assert os.path.isfile(model_config)
    return _UpstreamExpert(model_config, *args, **kwargs)
