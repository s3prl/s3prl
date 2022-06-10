from __future__ import annotations

import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import h5py
import numpy as np
import torch
import yaml
from filelock import FileLock

from .container import Container
from .object import Object

logger = logging.getLogger(__name__)


class FileHandler:
    @classmethod
    def save(cls, item, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError


class S3PRLObjectHandler(FileHandler):
    @classmethod
    def save(cls, item, path):
        assert isinstance(item, Object)
        item.save_checkpoint(str(path))

    @classmethod
    def load(cls, path):
        return Object.load_checkpoint(path)


class PickleHandler(FileHandler):
    @classmethod
    def save(cls, item, path):
        with open(path, "wb") as file:
            pickle.dump(item, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)


class TorchHandler(FileHandler):
    @classmethod
    def save(cls, item, path):
        torch.save(item, path)

    @classmethod
    def load(cls, path):
        return torch.load(path)


class NumpyHandler(FileHandler):
    @classmethod
    def save(cls, item, path):
        assert isinstance(item, np.ndarray)
        np.save(path, item)

    @classmethod
    def load(cls, path):
        return np.load(path)


class YamlHandler(FileHandler):
    @classmethod
    def save(cls, item, path):
        if isinstance(item, dict):
            item = Container(item).to_dict()
        with open(path, "w") as file:
            yaml.dump(item, file)

    @classmethod
    def load(cls, path):
        with open(path, "r") as file:
            item = yaml.load(file, Loader=yaml.FullLoader)
        return item


class StringHandler(FileHandler):
    @classmethod
    def save(cls, item, path):
        with open(path, "w") as file:
            file.write(repr(item))

    @classmethod
    def load(cls, path):
        with open(path, "r") as file:
            return eval(file.read())


class HDF5Handler(FileHandler):
    @classmethod
    def save(cls, item, path):
        with h5py.File(path, "w") as wf:
            wf.create_dataset("T_hat", data=item)

    @classmethod
    def load(cls, path):
        data = h5py.File(path, "r")
        return data["T_hat"][:]


class RTTMHandler(FileHandler):
    @classmethod
    def save(cls, all_segments: dict, path):
        assert isinstance(all_segments, dict)
        fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
        with open(path, "w") as wf:
            for recor, segments in all_segments.items():
                for spk, segs in segments.items():
                    for start, end in segs:
                        print(
                            fmt.format(
                                recor,
                                start,
                                end,
                                spk,
                            ),
                            file=wf,
                        )

    @classmethod
    def load(cls, path: str):
        output = dict()
        with open(path) as file:
            lines = [line.strip() for line in file.readlines()]
            for line in lines:
                line = re.sub(" +", " ", line)
                line = re.sub("\t+", " ", line)
                fields = line.split(" ")
                if fields[0] != "SPEAKER":
                    continue
                recor = fields[1]
                start = float(fields[3])
                end = float(fields[4])
                spk = fields[7]

                if recor not in output:
                    output[recor] = dict()

                if spk not in output[recor]:
                    output[recor][spk] = []

                output[recor][spk].append((start, end))

        return output


type_info = {
    "pkl": PickleHandler,
    "pt": TorchHandler,
    "npy": NumpyHandler,
    "yaml": YamlHandler,
    "obj": S3PRLObjectHandler,
    "txt": StringHandler,
    "h5": HDF5Handler,
    "rttm": RTTMHandler,
}


@dataclass
class TypeAssigner:
    item: Any
    ext: str


def as_type(obj: Any, ext: str):
    assert ext in type_info.keys()
    return TypeAssigner(obj, ext)


def save(filepath: str, obj: Any):
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)
    lockpath = filepath.parent / f".{filepath.stem}.lock"

    if isinstance(obj, TypeAssigner):
        ext = obj.ext
        obj = obj.item
    elif isinstance(obj, Object):
        ext = "obj"
    elif isinstance(obj, np.ndarray):
        ext = "npy"
    elif isinstance(obj, torch.Tensor):
        ext = "pt"
    else:
        ext = "pkl"

    if not str(filepath).endswith(f".{ext}"):
        filepath = Path(str(filepath) + f".{ext}")

    handler: FileHandler = type_info[ext]

    with FileLock(lockpath):
        handler.save(obj, filepath)


def load(filepath: str):
    ext = Path(filepath).suffix
    handler: FileHandler = type_info[ext.strip(".")]
    return handler.load(str(filepath))


def save_to_dir(root_dir: str, name: str, item: Any):
    rootdir = Path(root_dir)
    rootdir.mkdir(parents=True, exist_ok=True)

    ckpt_path = rootdir / f"{name}.ckpt"
    logger.info(
        f"Save '{name}' to {ckpt_path}. You can restore it by "
        f"{load_from_dir.__module__}.{load_from_dir.__qualname__}({str(rootdir)}, {name})"
    )
    save(ckpt_path, item)


def save_many_to_dir(root_dir: str, **name_item_pairs):
    for name, item in name_item_pairs.items():
        save_to_dir(root_dir, name, item)


def load_from_dir(root_dir: str, name: str, override: dict = None):
    rootdir = Path(root_dir)
    assert rootdir.is_dir()

    ckpt_path = rootdir / f"{name}.ckpt"
    assert ckpt_path.is_file(), str(ckpt_path)

    item = load(ckpt_path, override)
    return item


def load_many_from_dir(
    root_dir: str, names_and_overrides: Union[list, dict]
) -> Container:
    output = Container()
    if isinstance(names_and_overrides, (tuple, list)):
        names = names_and_overrides
        for name in names:
            output[name] = load_from_dir(root_dir, name)
    elif isinstance(names_and_overrides, dict):
        for name, override in names_and_overrides.items():
            output[name] = load_from_dir(root_dir, name, override)
    else:
        raise ValueError

    return output


def save_stats(root_dir: str, stats: dict, stats_name: str = "stats"):
    stats_dir = Path(root_dir) / stats_name
    stats_dir.mkdir(exist_ok=True, parents=True)
    for key, value in stats.items():
        save_to_dir(stats_dir, key, value)


def load_stats(root_dir: str, names: List[str] = None, stats_name: str = "stats"):
    stats_dir = Path(root_dir) / stats_name
    if not stats_dir.is_dir():
        return dict()

    if names is not None:
        for name in names:
            assert (stats_dir / f"{name}.ckpt").is_file()
    else:
        names = [Path(filename).stem for filename in os.listdir(str(stats_dir))]
        names = [name for name in names if not str(name).startswith(".")]

    output = dict()
    for name in names:
        output[name] = load_from_dir(stats_dir, name)

    return output
