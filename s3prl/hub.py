_available_options = []


def available_options():
    return _available_options.copy()


def _get_hubconf_entries():
    import importlib
    import logging
    import os
    import pathlib

    logger = logging.getLogger(__name__)

    _search_root = pathlib.Path(__file__).parent
    _hubconfs = list(_search_root.glob("upstream/*/hubconf.py"))

    for _hubconf in _hubconfs:
        relpath = _hubconf.relative_to(_search_root)
        try:
            _module_name = (
                "." + str(relpath).replace(os.path.sep, ".")[:-3]
            )  # remove .py
            _module = importlib.import_module(_module_name, package=__package__)

        except ModuleNotFoundError as e:
            if "pase" in _module_name:
                full_package = f"{__package__}{_module_name}"
                logger.warning(
                    f"Can not import {full_package}: {str(e)}. Please see {relpath.parent / 'README.md'}"
                )
                continue
            raise

        for variable_name in dir(_module):
            _variable = getattr(_module, variable_name)
            if callable(_variable) and variable_name[0] != "_":
                global _all_options
                _available_options.append(variable_name)

                globals()[variable_name] = _variable


_get_hubconf_entries()
del globals()["_get_hubconf_entries"]
