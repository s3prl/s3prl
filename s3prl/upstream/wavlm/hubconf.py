# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wavlm/hubconf.py ]
#   Synopsis     [ the WavLM torch hubconf ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os

# -------------#
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def wavlm_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def wavlm_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return wavlm_local(
        _urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs
    )


def wavlm(refresh=False, *args, **kwargs):
    """
    The default model - Base
        refresh (bool): whether to download ckpt/config again if existed
    """
    return wavlm_base_plus(refresh=refresh, *args, **kwargs)


def wavlm_base(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "\"https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Base.pt?sv=2020-04-08&st=2021-11-05T00%3A35%3A31Z&se=2022-11-06T00%3A35%3A00Z&sr=b&sp=r&sig=JljnRVzyHY6AjHzhVmHV5KyQQCvvGfgp9D2M02oGJBU%3D\""
    return wavlm_url(refresh=refresh, *args, **kwargs)


def wavlm_base_plus(refresh=False, *args, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "\"https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-04-08&st=2021-11-05T00%3A34%3A47Z&se=2022-10-06T00%3A34%3A00Z&sr=b&sp=r&sig=Gkf1IByHaIn1t%2FVEd9D6WHjZ3zu%2Fk5eSdoj21UytKro%3D\""
    return wavlm_url(refresh=refresh, *args, **kwargs)
