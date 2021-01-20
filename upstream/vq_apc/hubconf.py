from ..apc.hubconf import apc_local as vq_apc_local
from ..apc.hubconf import apc_url as vq_apc_url


def vq_apc(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return vq_apc_360hr(refresh=refresh, *args, **kwargs)


def vq_apc_360hr(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/6auicz4ovl0nwlq/vq_apc_default.ckpt?dl=0'
    return vq_apc_url(refresh=refresh, *args, **kwargs)
