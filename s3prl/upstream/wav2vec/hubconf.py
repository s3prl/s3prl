# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec/hubconf.py ]
#   Synopsis     [ the wav2vec torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def wav2vec_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): URL
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def wav2vec(refresh=False, *args, **kwargs):
    """
        The default model - Large model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec_large(refresh=refresh, *args, **kwargs)


def wav2vec_large(refresh=False, *args, **kwargs):
    """
        The Large model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt'
    return wav2vec_url(refresh=refresh, *args, **kwargs)