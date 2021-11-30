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


def wavlm_url(ckpt, refresh=False, agent="wget", *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return wavlm_local(
        _urls_to_filepaths(ckpt, refresh=refresh, agent=agent), *args, **kwargs
    )


def wavlm(refresh=False, *args, **kwargs):
    """
    The default model - Base-Plus
        refresh (bool): whether to download ckpt/config again if existed
    """
    return wavlm_base_plus(refresh=refresh, *args, **kwargs)


def wavlm_base(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    # Azure Storage
    kwargs["ckpt"] = "\"https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Base.pt?sv=2020-04-08&st=2021-11-05T00%3A35%3A31Z&se=2022-11-06T00%3A35%3A00Z&sr=b&sp=r&sig=JljnRVzyHY6AjHzhVmHV5KyQQCvvGfgp9D2M02oGJBU%3D\""

    # Google Drive
    # kwargs["ckpt"] = "https://drive.google.com/u/0/uc?id=19-C7SMQvEFAYLG5uc47NX_MY03JCbI4x&export=download"
    # kwargs["agent"] = 'gdown'
    return wavlm_url(refresh=refresh, *args, **kwargs)


def wavlm_base_plus(refresh=False, *args, **kwargs):
    """
    The Base-Plus model
        refresh (bool): whether to download ckpt/config again if existed
    """
    # Azure Storage
    kwargs["ckpt"] = "\"https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-04-08&st=2021-11-05T00%3A34%3A47Z&se=2022-10-06T00%3A34%3A00Z&sr=b&sp=r&sig=Gkf1IByHaIn1t%2FVEd9D6WHjZ3zu%2Fk5eSdoj21UytKro%3D\""

    # Google Drive
    # kwargs["ckpt"] = "https://drive.google.com/u/1/uc?id=1PlbT_9_B4F9BsD_ija84sUTVw7almNX8&export=download"
    # kwargs["agent"] = 'gdown'
    return wavlm_url(refresh=refresh, *args, **kwargs)


def wavlm_large(refresh=False, *args, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    # Azure Storage
    kwargs["ckpt"] = "\"https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Large.pt?sv=2020-08-04&st=2021-11-22T10%3A03%3A53Z&se=2022-11-23T10%3A03%3A00Z&sr=b&sp=r&sig=3kB8dwTCyIS8YQ7gW5oXmDrXV%2FAaLmoxBS37oPpFsz4%3D\""

    # Google Drive
    # kwargs["ckpt"] = "https://drive.google.com/u/1/uc?id=1p8nbj16b7YA16sqPZ4E0JUL-oIDUBGwU&export=download"
    # kwargs["agent"] = 'gdown'
    return wavlm_url(refresh=refresh, *args, **kwargs)