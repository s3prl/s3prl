import os as _os
import pathlib as _pathlib
import importlib as _importlib

_search_root = _pathlib.Path(__file__).parent
_hubconfs = list(_search_root.glob("upstream/*/hubconf.py"))
_hubconfs += list(_search_root.glob("downstream/*/hubconf.py"))

for _hubconf in _hubconfs:
    relpath = _hubconf.relative_to(_search_root)
    try:
        _module_name = "." + str(relpath).replace(_os.path.sep, ".")[:-3]  # remove .py
        _module = _importlib.import_module(_module_name, package=__package__)

    except ModuleNotFoundError as e:
        print(f'[Warning] can not import {_module_name}: {str(e)}. Please see {relpath.parent / "README.md"}')
        continue

    for variable_name in dir(_module):
        _variable = getattr(_module, variable_name)
        if callable(_variable) and variable_name[0] != '_':
            globals()[variable_name] = _variable
