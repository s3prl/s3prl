# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/unispeech_sat/hubconf.py ]
#   Synopsis     [ the UniSpeech-SAT torch hubconf ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os

# -------------#
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def unispeech_sat_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def unispeech_sat_url(ckpt, refresh=False, agent="wget", *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return unispeech_sat_local(
        _urls_to_filepaths(ckpt, refresh=refresh, agent=agent), *args, **kwargs
    )


def unispeech_sat(refresh=False, *args, **kwargs):
    """
    The default model - Base-Plus
        refresh (bool): whether to download ckpt/config again if existed
    """
    return unispeech_sat_base_plus(refresh=refresh, *args, **kwargs)


def unispeech_sat_base(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    # Azure Storage
    kwargs["ckpt"] = "\"https://msranlcmtteamdrive.blob.core.windows.net/share/unispeech-sat/UniSpeech-SAT-Base.pt?sv=2020-08-04&st=2021-11-26T10%3A09%3A40Z&se=2022-11-27T10%3A09%3A00Z&sr=b&sp=r&sig=iHQ9HTPwajdzHPVXAsYWeYFbDQ4a%2BmVdpL9BpoKBa5g%3D\""

    # Google Drive
    # kwargs["ckpt"] = "https://drive.google.com/u/1/uc?id=1j6WMIdOIu_GMtRVINTqjxMsHq_cf98_b&export=download"
    # kwargs["agent"] = 'gdown'
    return unispeech_sat_url(refresh=refresh, *args, **kwargs)


def unispeech_sat_base_plus(refresh=False, *args, **kwargs):
    """
    The Base-Plus model
        refresh (bool): whether to download ckpt/config again if existed
    """
    # Azure Storage
    kwargs["ckpt"] = "\"https://msranlcmtteamdrive.blob.core.windows.net/share/unispeech-sat/UniSpeech-SAT-Base+.pt?sv=2020-08-04&st=2021-11-26T10%3A10%3A25Z&se=2022-11-27T10%3A10%3A00Z&sr=b&sp=r&sig=plC0%2BNmN7Q18RgdmHBIrEBK2IuMUSW%2F0AnLqleO2JX8%3D\""

    # Google Drive
    # kwargs["ckpt"] = "https://drive.google.com/u/1/uc?id=1AymTVpum41nMlGQLqheRO_kaFKbxZvvV&export=download"
    # kwargs["agent"] = 'gdown'
    return unispeech_sat_url(refresh=refresh, *args, **kwargs)


def unispeech_sat_large(refresh=False, *args, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    # Azure Storage
    kwargs["ckpt"] = "\"https://msranlcmtteamdrive.blob.core.windows.net/share/unispeech-sat/UniSpeech-SAT-Large.pt?sv=2020-08-04&st=2021-11-26T10%3A10%3A59Z&se=2022-11-27T10%3A10%3A00Z&sr=b&sp=r&sig=U7cExvz%2Bt4mVGdN9mdRQ0U%2FodUuS25wGcHtoUmk2Dd4%3D\""

    # Google Drive
    # kwargs["ckpt"] = "https://drive.google.com/u/1/uc?id=15FR4Y1vohoVnOTc_ob7K9OUn0L1KrV1A&export=download"
    # kwargs["agent"] = 'gdown'
    return unispeech_sat_url(refresh=refresh, *args, **kwargs)