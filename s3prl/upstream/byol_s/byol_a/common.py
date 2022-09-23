"""BYOL for Audio: Common definitions and utilities."""
import datetime
import logging
import os
import random
from argparse import Namespace
from pathlib import Path

import yaml

try:
    import pickle5 as pickle
except ImportError:
    import pickle

import numpy as np
import torch
import torchaudio

torchaudio.set_audio_backend("sox_io")


def get_timestamp():
    """ex) Outputs 202104220830"""
    return datetime.datetime.now().strftime("%y%m%d%H%M")


def load_yaml_config(path_to_config):
    """Loads yaml configuration settings as an EasyDict object."""
    path_to_config = Path(path_to_config)
    assert path_to_config.is_file()
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    cfg = Namespace(**yaml_contents)
    return cfg


def get_logger(name):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.DEBUG,
    )
    logger = logging.getLogger(name)
    return logger
