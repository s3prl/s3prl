# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol_a/hubconf.py ]
#   Synopsis     [ the BYOL-A torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

from pathlib import Path as _Path

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

DEFAULT_CONFIG_PATH = _Path(__file__).parent / "config.yaml"


def _byol_a(refresh=False, **kwargs):
    assert 'ckpt' in kwargs, '*** Please maks sure to set "-k your-checkpoint.pth".'
    if 'model_config' not in kwargs or not kwargs['model_config']:
        kwargs['model_config'] = DEFAULT_CONFIG_PATH
    return _UpstreamExpert(**kwargs)


def byol_a_calcnorm(refresh=False, **kwargs):
    """Calculate downstream task statistics using this model."""
    return _byol_a(**kwargs)


def byol_a(refresh=False, *args, **kwargs):
    """
    kwargs['ckpt'] is expected in the format of "path-of-ckpt,dataset-mean,dataset-std".
    """
    assert 'ckpt' in kwargs, '*** Please maks sure to set "-k your-checkpoint.pth,dataset-mean,dataset-std".'
    if kwargs['ckpt'] is None:
        print('Set "-k your-checkpoint.pth,norm_mean,norm_std". Exit now.')
        exit(-1)
    ckpt, norm_mean, norm_std = kwargs['ckpt'].split(',')
    kwargs['ckpt'] = ckpt
    norm_mean, norm_std = float(norm_mean), float(norm_std)
    print(' using checkpoint:', ckpt)
    print(' normalization statistics:', norm_mean, norm_std)
    return _byol_a(*args, norm_mean=norm_mean, norm_std=norm_std, **kwargs)
