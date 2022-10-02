"""
Parse command-line arguments into override dictionary

Authors
  * Leo 2022
"""

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "parse_overrides",
]


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


def parse_overrides(options: list):
    """
    Example usgae:
        [
            "--optimizer.lr",
            "1.0e-3",
            "--optimizer.name",
            "AdamW",
            "--runner.eval_dataloaders",
            "['dev', 'test']",
        ]

    Convert to:
        {
            "optimizer": {"lr": 1.0e-3, "name": "AdamW"},
            "runner": {"eval_dataloaders": ["dev", "test"]}
        }
    """
    config = {}
    for position in range(0, len(options), 2):
        key: str = options[position]
        assert key.startswith("--")
        key = key.strip("--")
        value_str: str = options[position + 1]
        key, value_str = key.strip(), value_str.strip()
        remaining = key.split(".")

        try:
            value = eval(value_str)
        except Exception as e:
            if "newdict" in value_str or "Container" in value_str:
                raise
            value = value_str

        logger.debug(f"{key} = {value}")

        target_config = config
        for i, field_name in enumerate(remaining):
            if i == len(remaining) - 1:
                target_config[field_name] = value
            else:
                target_config.setdefault(field_name, {})
                target_config = target_config[field_name]
    return config
