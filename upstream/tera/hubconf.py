import os
import torch

from utility.download import _gdriveids_to_filepaths
from upstream.tera.expert import UpstreamExpert as _UpstreamExpert


def tera(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def tera_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return tera(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def tera_default(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = '1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_fbankBase_time_100hr(refresh=False, *args, **kwargs):
    kwargs['ckpt'] = '1zS_r-24ir81f_SrEW5-qvKN9DNog-yEV'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_fbankBase_freq_100hr(refresh=False, *args, **kwargs):
    kwargs['ckpt'] = '1CpxVvdTQpRA3iIDWpa7HFjPFzzm7YeTZ'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)

    
def tera_fbankBase_time_freq_100hr(refresh=False, *args, **kwargs):
    kwargs['ckpt'] = '1gaLkpG9knX64kcawdc5R7VazgOz25QKx'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_fbankBase_time_freq_mag_100hr(refresh=False, *args, **kwargs):
    kwargs['ckpt'] = '1CjGPGdrg66OrM8DcpK5pThE8p8OsU8bA'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_fbankBase_time_mag_100hr(refresh=False, *args, **kwargs):
    kwargs['ckpt'] = 'todo'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_fbankBase_freq_mag_100hr(refresh=False, *args, **kwargs):
    kwargs['ckpt'] = 'todo'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)


def tera_fbankBase_mag_100hr(refresh=False, *args, **kwargs):
    kwargs['ckpt'] = 'todo'
    return tera_gdriveid(refresh=refresh, *args, **kwargs)