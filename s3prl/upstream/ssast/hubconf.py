# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : hubconf.py

# Authors
# - Leo

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


# Frame-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssast_frame_base(refresh: bool = False, window_secs: float = 1.0, **kwargs):
    ckpt = _urls_to_filepaths(
        "https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1",
        refresh=refresh,
    )
    return _UpstreamExpert(ckpt, "base_f", window_secs)


# Patch-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssast_patch_base(refresh: bool = False, window_secs: float = 1.0, **kwargs):
    ckpt = _urls_to_filepaths(
        "https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1",
        refresh=refresh,
    )
    return _UpstreamExpert(ckpt, "base_p", window_secs)
