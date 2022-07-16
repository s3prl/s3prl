# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/spec_augment/hubconf.py ]
#   Synopsis     [ the spec augment torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

import torch

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def spec_augment_local(ckpt, options_config=None, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, options_config=options_config, *args, **kwargs)


def spec_augment_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return spec_augment_local(
        _urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs
    )


def spec_augment(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://www.dropbox.com/s/spz3yulaye8ppgr/states-100000.ckpt?dl=1"
    return spec_augment_url(refresh=refresh, *args, **kwargs)
