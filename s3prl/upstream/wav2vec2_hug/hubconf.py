from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec2_hug(ckpt, *args, **kwargs):
    """
        ckpt:
            The identifier string of huggingface wav2vec2 models.
            eg. facebook/wav2vec2-base-960h
            see https://huggingface.co/facebook
    """

    return _UpstreamExpert(ckpt, *args, **kwargs)


def wav2vec2_hug_base_960(*args, **kwargs):
    kwargs['ckpt'] = 'facebook/wav2vec2-base'
    return wav2vec2_hug(*args, **kwargs)


def wav2vec2_hug_large_ll60k(*args, **kwargs):
    kwargs['ckpt'] = 'facebook/wav2vec2-large-lv60'
    return wav2vec2_hug(*args, **kwargs)


def wav2vec2_hug_large_lv60_self_training(*args, **kwargs):
    kwargs['ckpt'] = 'facebook/wav2vec2-large-960h-lv60-self'
    return wav2vec2_hug(*args, **kwargs)