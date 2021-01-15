from ..mockingjay.hubconf import mockingjay_local as audio_albert_local
from ..mockingjay.hubconf import mockingjay_gdriveid as audio_albert_gdriveid

def audio_albert(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    raise NotImplementedError
    kwargs['ckpt'] = 'TBD'
    return audio_albert_gdriveid(refresh=refresh, *args, **kwargs)
