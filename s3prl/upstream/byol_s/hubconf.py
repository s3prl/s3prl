from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def byol_s_default(refresh: bool = False, **kwds):
    kwds["model_name"] = "default"
    kwds["ckpt"] = _urls_to_filepaths(
        "https://github.com/GasserElbanna/serab-byols/raw/main/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth",
        refresh=refresh,
    )
    return _UpstreamExpert(**kwds)


def byol_s_cvt(refresh: bool = False, **kwds):
    kwds["model_name"] = "cvt"
    kwds["ckpt"] = _urls_to_filepaths(
        "https://github.com/GasserElbanna/serab-byols/raw/main/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth",
        refresh=refresh,
    )
    return _UpstreamExpert(**kwds)


def byol_s_resnetish34(refresh: bool = False, **kwds):
    kwds["model_name"] = "resnetish34"
    kwds["ckpt"] = _urls_to_filepaths(
        "https://github.com/GasserElbanna/serab-byols/raw/main/checkpoints/resnetish34_BYOLAs64x96-2105271915-e100-bs256-lr0003-rs42.pth",
        refresh=refresh,
    )
    return _UpstreamExpert(**kwds)
