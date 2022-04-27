import importlib
import logging

logger = logging.getLogger(__name__)


def qualname_to_cls(qualname: str):
    module_name, cls_name = qualname.rsplit(".", maxsplit=1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls


def parse_override(string):
    """
    Example usgae:
        -o "optimizer.lr=1.0e-3,,optimizer.name='AdamW',,runner.eval_dataloaders=['dev', 'test']"

    Convert to:
        {
            "optimizer": {"lr": 1.0e-3, "name": "AdamW"},
            "runner": {"eval_dataloaders": ["dev", "test"]}
        }
    """
    options = string.split(",,")
    config = {}
    for option in options:
        option = option.strip()
        key, value_str = option.split("=")
        key, value_str = key.strip(), value_str.strip()
        remaining = key.split(".")

        try:
            value = eval(value_str)
        except:
            value = value_str

        logger.info(f"{key} = {value}")

        target_config = config
        for i, field_name in enumerate(remaining):
            if i == len(remaining) - 1:
                target_config[field_name] = value
            else:
                target_config.setdefault(field_name, {})
                target_config = target_config[field_name]
    return config
