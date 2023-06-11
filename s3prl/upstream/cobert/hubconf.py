from .expert import UpstreamExpert as _UpstreamExpert
from s3prl.util.download import urls_to_filepaths


def cobert(**kwargs):
    kwargs["ckpt"] = urls_to_filepaths("https://huggingface.co/s3prl/cobert/resolve/main/cobert.pt")
    return _UpstreamExpert(**kwargs)
