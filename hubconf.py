# -*- coding: utf-8 -*- #
"""************************************************************************************************
   FileName     [ hubconf.py ]
   Synopsis     [ interface to Pytorch Hub: https://pytorch.org/docs/stable/hub.html#torch-hub ]
   Author       [ S3PRL ]
   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
************************************************************************************************"""

import os
import hashlib
import pathlib
import importlib

import torch
dependencies = ['torch']


search_root = os.path.dirname(__file__)
hubconfs = [str(p) for p in pathlib.Path(search_root).rglob('hubconf.py')]
hubconfs = [os.path.relpath(p, search_root) for p in hubconfs]
hubconfs.remove('hubconf.py')  # remove the root hubconf.py

for hubconf in hubconfs:
    module_name = '.'.join(str(hubconf).split('.')[:-1]).replace('/', '.')
    try:
        _module = importlib.import_module(module_name)
    except RuntimeError as e:
        print(str(e))
        print(f'[hubconf] import error - {module_name}')
        continue

    for variable_name in dir(_module):
        _variable = getattr(_module, variable_name)
        if callable(_variable) and variable_name[0] != '_':
            globals()[variable_name] = _variable
