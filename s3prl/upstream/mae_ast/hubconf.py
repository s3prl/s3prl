# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mae_ast/hubconf.py ]
#   Synopsis     [ the MAE-AST torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def mae_ast_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def mae_ast_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return mae_ast_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def mae_ast(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    return mae_ast_frame(refresh=refresh, *args, **kwargs)


def mae_ast_frame(refresh=False, *args, **kwargs):
    """
    The MAE-AST Frame model, 12-layered, random masking
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://www.cs.utexas.edu/~harwath/model_checkpoints/mae_ast/random_frame_75_12LayerEncoder.pt"
    return mae_ast_url(refresh=refresh, *args, **kwargs)


def mae_ast_patch(refresh=False, *args, **kwargs):
    """
    The MAE-AST Patch model, 12-layered, chunked masking
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://www.cs.utexas.edu/~harwath/model_checkpoints/mae_ast/chunk_patch_75_12LayerEncoder.pt"
    return mae_ast_url(refresh=refresh, *args, **kwargs)
