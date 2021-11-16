from .expert import UpstreamExpert as _UpstreamExpert


def unispeech_hug(ckpt, *args, **kwargs):
    """
        ckpt:
            The identifier string of huggingface unispeech models.
            eg. microsoft/unispeech-sat-base
            see https://huggingface.co/microsoft
    """

    return _UpstreamExpert(ckpt, *args, **kwargs)


def unispeechsat_hug_base(*args, **kwargs):
    kwargs['ckpt'] = 'microsoft/unispeech-sat-base'
    return unispeech_hug(*args, **kwargs)


def unispeechsat_hug_base_plus(*args, **kwargs):
    kwargs['ckpt'] = 'microsoft/unispeech-sat-base-plus'
    return unispeech_hug(*args, **kwargs)


def unispeechsat_hug_large(*args, **kwargs):
    kwargs['ckpt'] = 'microsoft/unispeech-sat-large'
    return unispeech_hug(*args, **kwargs)
