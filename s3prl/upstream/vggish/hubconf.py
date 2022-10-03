# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/vggish/hubconf.py ]
#   Synopsis     [ the VGGish torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import torch

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def _load_state_dict_from_url(url):
    path = _urls_to_filepaths(url)
    return torch.load(path, map_location="cpu")


def _vggish_from_torch_hub(urls, *args, **kwargs):
    """
    The model from `torch.hub.load`
        urls (dict): LINKS
    """
    kwargs["ckpt"] = {
        "vggish": _load_state_dict_from_url(urls["vggish"]),
        "pca": _load_state_dict_from_url(urls["pca"]),
    }
    return _UpstreamExpert(*args, **kwargs)


def vggish(*args, **kwargs):
    """
    The default model
    """
    urls = {
        "vggish": "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth",
        "pca": "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth",
    }
    return _vggish_from_torch_hub(urls, *args, **kwargs)
