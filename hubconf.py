# -*- coding: utf-8 -*- #
"""************************************************************************************************
   FileName     [ hubconf.py ]
   Synopsis     [ interface to Pytorch Hub: https://pytorch.org/docs/stable/hub.html#torch-hub ]
   Author       [ S3PRL ]
   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
************************************************************************************************"""

import pathlib as _pathlib
import importlib as _importlib

_search_root = _pathlib.Path(__file__).parent
_hubconfs = _search_root.glob("s3prl/*/*/hubconf.py")

for _hubconf in _hubconfs:
    posixpath = _hubconf.relative_to(_search_root).as_posix()
    try:
        _module_name = ".".join(posixpath.split(".")[:-1]).replace("/", ".")
        _module = _importlib.import_module(_module_name)

    except ModuleNotFoundError as e:
        print(f'[hubconf] can not import {_module_name}: {str(e)}... Pass.')
        continue

    for variable_name in dir(_module):
        _variable = getattr(_module, variable_name)
        if callable(_variable) and variable_name[0] != '_':
            globals()[variable_name] = _variable
