# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/pase/hubconf.py ]
#   Synopsis     [ the pase torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

import torch

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def pase_local(ckpt, model_config, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
        model_config (str): PATH
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(model_config)
    return _UpstreamExpert(ckpt, model_config, **kwargs)


def pase_url(ckpt, model_config, refresh=False, **kwargs):
    """
    The model from URL
        ckpt (str): URL
        model_config (str): URL
    """
    ckpt = _urls_to_filepaths(ckpt, refresh=refresh)
    model_config = _urls_to_filepaths(model_config, refresh=refresh)
    return pase_local(ckpt, model_config, **kwargs)


def pase_plus(refresh=False, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://www.dropbox.com/s/p8811o7eadv4pat/FE_e199.ckpt?dl=1"
    kwargs[
        "model_config"
    ] = "https://www.dropbox.com/s/2p3ouod1k0ekfxn/PASE%2B.cfg?dl=1"

    def align_skip(input_, skip):
        """
        Ref: https://github.com/s3prl/pase/blob/be11486c907db4bd2887ba96d656edc3f8fffec4/pase/models/frontend.py#L213
        """
        dfactor = skip.shape[2] // input_.shape[2]
        if dfactor > 1:
            maxlen = input_.shape[2] * dfactor
            skip = skip[:, :, :maxlen]
            bsz, feats, slen = skip.shape
            skip_re = skip.view(bsz, feats, slen // dfactor, dfactor)
            skip = torch.mean(skip_re, dim=3)
        return skip

    from typing import List, Tuple

    from torch import Tensor

    def hook_postprocess(hiddens: List[Tuple[str, Tensor]]):
        remained_hiddens = [x for x in hiddens if x[0] != "self.model"]
        final_hidden = [x for x in hiddens if x[0] == "self.model"]
        assert len(final_hidden) == 1
        final_hidden = final_hidden[0]

        updated_hiddens = []
        for identifier, tensor in remained_hiddens:
            updated_hiddens.append(
                (identifier, align_skip(final_hidden[1], tensor).transpose(1, 2))
            )

        updated_hiddens.append((final_hidden[0], final_hidden[1].transpose(1, 2)))
        return updated_hiddens

    hooks = [
        (f"self.model.W", lambda input, output: output),
        (f"self.model", lambda input, output: output),
    ]
    for i in range(7):
        hooks.append((f"self.model.denseskips[{i}]", lambda input, output: output))

    kwargs["hooks"] = hooks
    kwargs["hook_postprocess"] = hook_postprocess
    return pase_url(refresh=refresh, **kwargs)
