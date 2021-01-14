from ..apc.hubconf import apc_local as vq_apc_local
from ..apc.hubconf import apc_gdriveid as vq_apc_gdriveid


def vq_apc(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = '1MlrHTKIA1HHs8FPbMN64f0O-7wWMBthF'
    return vq_apc_gdriveid(refresh=refresh, *args, **kwargs)
