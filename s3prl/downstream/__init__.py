import pathlib as _pathlib
import importlib as _importlib

_search_root = _pathlib.Path(__file__).parent
for _subdir in _search_root.iterdir():
    if _subdir.is_dir() and (_subdir / "expert.py").is_file():
        _name = str(_subdir.relative_to(_search_root))
        try:
            _module_name = f".{_name}.expert"
            _module = _importlib.import_module(_module_name, package=__package__)

        except ModuleNotFoundError as e:
            print(f'[Downstream] can not import {_module_name}: {str(e)}... Pass.')
            continue

        globals()[_name] = getattr(_module, "DownstreamExpert")
