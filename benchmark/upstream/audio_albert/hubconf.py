from utility.download import _gdriveids_to_filepaths

from ..mockingjay.hubconf import mockingjay as audio_albert
from ..mockingjay.hubconf import mockingjay_gdriveid as audio_albert_gdriveid


def audio_albert_default(refresh=False, *args, **kwargs):
    f"""
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    raise NotImplementedError
    kwargs['ckpt'] = 'TBD'
    return audio_albert_gdriveid(refresh=refresh, *args, **kwargs)
