from .expert import UpstreamExpert as _UpstreamExpert
from s3prl.util.download import urls_to_filepaths


def dp_hubert(**kwargs):
    kwargs["ckpt"] = urls_to_filepaths("https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth")
    return _UpstreamExpert(**kwargs)


def dp_wavlm(**kwargs):
    kwargs["ckpt"] = urls_to_filepaths("https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPWavLM-sp0.75.pth")
    return _UpstreamExpert(**kwargs)
