import importlib


def top_level_package_version(cls):
    module_name = cls.__module__
    top_package_name = module_name.split(".")[0]
    top_package = importlib.import_module(top_package_name)
    return getattr(top_package, "__version__", None)
