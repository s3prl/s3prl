# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec2_ft/hubconf.py ]
#   Synopsis     [ the wav2vec 2.0 fine-tuned torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
#-------------#
from utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec2_ft_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def wav2vec2_ft_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): URL
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec2_ft_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def wav2vec2_ft(refresh=False, *args, **kwargs):
    """
        The default model - Base
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt'
    return wav2vec2_ft_url(refresh=refresh, *args, **kwargs)
