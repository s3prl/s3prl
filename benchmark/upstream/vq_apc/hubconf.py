from ..apc.hubconf import apc as vq_apc
from ..apc.hubconf import apc_gdriveid as vq_apc_gdriveid


def vq_apc_default(refresh=False, *args, **kwargs):
    f"""
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    ckpt = '1swpF6nCLU2xVWRmwbt0s2w0BkQ8ys2iy'
    config = '1aH13mvds_ZERD5pI0NfOTBtFhbKQL_0_'
    return vq_apc_gdriveid(ckpt, config, refresh=refresh)
