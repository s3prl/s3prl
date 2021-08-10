import pathlib as _pathlib
import importlib as _importlib

_search_root = _pathlib.Path(__file__).parent
_hubconfs = _search_root.glob("*/*/hubconf.py")

for _hubconf in _hubconfs:
    posixpath = _hubconf.relative_to(_search_root).as_posix()
    try:
        _module_name = "." + ".".join(posixpath.split(".")[:-1]).replace("/", ".")
        _module = _importlib.import_module(_module_name, package=__package__)

    except ModuleNotFoundError as e:
        print(f'[hubconf] can not import {_module_name}: {str(e)}... Pass.')
        continue

    for variable_name in dir(_module):
        _variable = getattr(_module, variable_name)
        if callable(_variable) and variable_name[0] != '_':
            globals()[variable_name] = _variable
