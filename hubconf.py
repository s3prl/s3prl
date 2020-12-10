# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ hubconf.py ]
#   Synopsis     [ interface to Pytorch Hub: https://pytorch.org/docs/stable/hub.html#torch-hub ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

import torch
from transformer.nn_transformer import TRANSFORMER as _TRANSFORMER
dependencies = ['torch', 'torchaudio', 'numpy']


options = {'load_pretrain' : 'True',
           'no_grad'       : 'True',
           'dropout'       : 'default',
           'spec_aug'      : 'False',
           'spec_aug_prev' : 'True',
           'weighted_sum'  : 'False',
           'select_layer'  : -1,
           'permute_input' : 'False' }


# mockingjay is the name of entrypoint
def mockingjay(ckpt=None, **kwargs):
    """
    The Mockingjay model
    ckpt (str): kwargs, path to the pretrained weights of the model,
                if None is given a default ckpt will be downloaded.
    Usage:
        >>> repo = 'andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning'
        >>> path2ckpt = 'on-the-fly-melBase960-b12-d01-T-libri.ckpt'
        >>> model = torch.hub.load(repo, 'mockingjay', ckpt=path2ckpt)
    """

    if ckpt is None:
        ckpt_url = 'https://drive.google.com/u/1/uc?id=1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH&export=download'
        ckpt = torch.hub.load_state_dict_from_url(ckpt_url, progress=True)
    else:
        ckpt = torch.load(ckpt, map_location='cpu')

    options['ckpt_file'] = ckpt
    model = _TRANSFORMER(options, inp_dim=-1)
    return model


# tera is the name of entrypoint
def tera(ckpt=None, **kwargs):
    """
    The TERA model
    ckpt (str): kwargs, path to the pretrained weights of the model,
                if None is given a default ckpt will be downloaded.
    Usage:
        >>> repo = 'andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning'
        >>> path2ckpt = 'on-the-fly-melBase960-b128-d03-T-C-libri.ckpt'
        >>> model = torch.hub.load(repo, 'mockingjay', ckpt=path2ckpt)
    """
    
    if ckpt is None:
        ckpt_url = 'https://drive.google.com/u/1/uc?id=1A9Fs2k3aekY4_6I2GD4tBtjx_v0mV_k4&export=download'
        ckpt = torch.hub.load_state_dict_from_url(ckpt_url, progress=True)
    else:
        ckpt = torch.load(ckpt, map_location='cpu')

    options['ckpt_file'] = ckpt
    model = _TRANSFORMER(options, inp_dim=-1)
    return model


# audio_albert is the name of entrypoint
def audio_albert(ckpt=None, **kwargs):
    """
    The Audio ALBERT model
    ckpt (str): kwargs, path to the pretrained weights of the model,
                if not given a default ckpt will be downloaded.
    Usage:
        >>> repo = 'andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning'
        >>> path2ckpt = 'todo.ckpt'
        >>> model = torch.hub.load(repo, 'mockingjay', ckpt=path2ckpt)
    """

    if ckpt is None:
        ckpt_url = 'https://drive.google.com/u/1/uc?id=todo&export=download'
        ckpt = torch.hub.load_state_dict_from_url(ckpt_url, progress=True)
    else:
        ckpt = torch.load(ckpt, map_location='cpu')

    options['ckpt_file'] = ckpt
    model = _TRANSFORMER(options, inp_dim=-1)
    return model
