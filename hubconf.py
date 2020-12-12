# -*- coding: utf-8 -*- #
"""************************************************************************************************
   FileName     [ hubconf.py ]
   Synopsis     [ interface to Pytorch Hub: https://pytorch.org/docs/stable/hub.html#torch-hub ]
   Author       [ S3PRL ]
   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
************************************************************************************************"""

import os
import importlib

import torch
import gdown
dependencies = ['torch', 'gdown']


def _gdown(filename, url, use_cache):
    filepath = f'{torch.hub.get_dir()}/{filename}'
    if not os.path.isfile(filepath) or not use_cache:
        print(f'Downloading file to {filepath}')
        gdown.download(url, filepath, use_cookies=False)
    else:
        print(f'Using cache found in {filepath}')
    return filepath


for upstream_dir in os.listdir('benchmark/upstream'):
    hubconf_path = os.path.join('benchmark/upstream', upstream_dir, 'hubconf.py')
    if os.path.isfile(hubconf_path):
        module_path = f'benchmark.upstream.{upstream_dir}.hubconf'
        _module = importlib.import_module(module_path)
        for variable_name in dir(_module):
            _variable = getattr(_module, variable_name)
            if callable(_variable) and variable_name[0] != '_':
                globals()[variable_name] = _variable
