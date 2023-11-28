# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/multires_hubert/hubconf.py ]
#   Synopsis     [ the Multiresolution HuBERT torch hubconf ]
#   Author       [ S3PRL / Jiatong Shi ]
"""*********************************************************************************************"""


import logging
import os
import time
from pathlib import Path

from filelock import FileLock
from s3prl.util.download import _urls_to_filepaths

from .convert import load_and_convert_fairseq_ckpt
from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)

NEW_ENOUGH_SECS = 2.0


def multires_hubert_custom(
    ckpt: str,
    refresh: bool = False,
    **kwargs,
):

    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt=ckpt, **kwargs)


def multires_hubert_local(*args, **kwargs):
    return multires_hubert_custom(*args, **kwargs)
