# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/pase/hubconf.py ]
#   Synopsis     [ the pase torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os
import torch

from utility.download import _urls_to_filepaths
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
    kwargs['ckpt'] = 'https://www.dropbox.com/s/p8811o7eadv4pat/FE_e199.ckpt?dl=0'
    kwargs['model_config'] = 'https://www.dropbox.com/s/2p3ouod1k0ekfxn/PASE%2B.cfg?dl=0'

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
    
    def hook_postprocess(hook_hiddens):
        transformed = {}
        final_hidden = hook_hiddens.pop("self.model")
        for module_name, hidden in hook_hiddens.items():
            transformed[module_name] = align_skip(final_hidden, hidden).transpose(1, 2)
        transformed["self.model"] = final_hidden.transpose(1, 2)
        return transformed

    hooks = {
        f"self.model.W": lambda input, output: output,
        f"self.model": lambda input, output: output,        
    }
    for i in range(7):
        hooks[f"self.model.denseskips[{i}]"] = lambda input, output: output

    kwargs["hooks"] = hooks
    kwargs["hook_postprocess"] = hook_postprocess
    return pase_url(refresh=refresh, **kwargs)
