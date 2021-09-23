def _get_experts():
    import pathlib
    import importlib

    _search_root = pathlib.Path(__file__).parent
    for _subdir in _search_root.iterdir():
        if _subdir.is_dir() and (_subdir / "expert.py").is_file():
            _name = str(_subdir.relative_to(_search_root))
            try:
                _module_name = f".{_name}.expert"
                _module = importlib.import_module(_module_name, package=__package__)

            except ModuleNotFoundError as e:
                if "pase" in _module_name:
                    # pase is not installed by default. See upstream/pase/README.md
                    continue

                full_package = f"{__package__}{_module_name}"
                print(f'[{__name__}] Warning: can not import {full_package}: {str(e)}. Pass.')
                continue
            
            globals()[_name] = getattr(_module, "UpstreamExpert")

_get_experts()
del globals()["_get_experts"]
