# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : hubconf.py

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


# Frame-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssast_frame_base(*args, segment_secs: float = 1.0, **kwargs):
    kwargs["model_size"] = "base_f"
    kwargs["pretrain_path"] = _urls_to_filepaths(
        "https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1"
    )
    kwargs["target_length"] = round(segment_secs * 100)
    return _UpstreamExpert(*args, **kwargs)


# FIXME: Enable Patch-based
# Patch-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
# def ssast_patch_base(*args, mode: str = "1.0_sec", **kwargs):
#     kwargs["model_size"] = "base_p"
#     kwargs["pretrain_path"] = _urls_to_filepaths("https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1")
#     kwargs["target_length"] = round(segment_secs * 100)
#     return _UpstreamExpert(*args, **kwargs)
