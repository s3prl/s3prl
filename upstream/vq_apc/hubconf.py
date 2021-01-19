from ..apc.hubconf import apc_local as vq_apc_local
from ..apc.hubconf import apc_url as vq_apc_url


def vq_apc(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'http://140.112.21.12:8000/vqapc/vqapc_360hr.ckpt'
    return vq_apc_url(refresh=refresh, *args, **kwargs)
