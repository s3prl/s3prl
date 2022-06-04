from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from types import MethodType
from typing import Any

from s3prl.base.container import Container, field
from s3prl.util.checkpoint import as_type, load, save

logger = logging.getLogger(__name__)


class Workspace(type(Path()), MutableMapping):
    def __new__(cls, path=None):
        if path is None:
            path = tempfile.mkdtemp()
            is_temp = True
        else:
            is_temp = False
        obj = super().__new__(cls, path)
        obj._is_temp = is_temp
        obj._rank = 0
        obj._default_dtype = None
        return obj

    def __truediv__(self, key):
        workspace = __class__(super().__truediv__(key))
        workspace.set_rank(self._rank)
        return workspace

    def __repr__(self):
        return f"{__class__.__name__}('{str(self)}', rank={self._rank})"

    def __del__(self):
        if getattr(self, "_is_temp", False):
            shutil.rmtree(str(self))

    def set_rank(self, rank: int):
        assert isinstance(rank, int)
        self._rank = rank

    def set_default_dtype(self, dtype: str):
        self._default_dtype = dtype

    @property
    def rank(self):
        return self._rank

    @staticmethod
    def _find_file(parent_dir: str, stem):
        parent_dir = Path(parent_dir)
        stems = [Path(item).stem for item in os.listdir(str(parent_dir))]
        assert len(set(stems)) == len(stems), (
            f"There are duplicated keys in {str(parent_dir)}. "
            "Might be caused by the same filename while different extensions (dtype), "
            "or a directory has the same name as a file's stem. This is considered as "
            "a bad practice in S3PRL, since this can cause confusion when distributing "
            "the workspace (as the checkpoint for the entire experiment), it can be hard "
            "to understand the file structure without tracing the code if the filenames "
            "are not uniquely defined. Please make sure that each key in a workspace only "
            "has an unique corresponding file or directory"
        )
        stem2file = {Path(item).stem: item for item in os.listdir(str(parent_dir))}
        return stem2file.get(stem)

    def get_filepath(self, identifier):
        """
        Get an object
        """
        try:
            filename = self._find_file(self, identifier)
        except FileNotFoundError:
            filename = None

        if filename is not None:
            filepath = (self / filename).resolve()
            if len(filepath.suffix) > 0:
                return filepath
        return None

    def get(self, *identifier_and_default, get_filepath=False) -> Any:
        """
        Get an object
        """
        identifier = identifier_and_default[0]
        if len(identifier_and_default) > 2:
            raise ValueError("get takes at most 2 arguments")

        filepath = self.get_filepath(identifier)
        if filepath is not None:
            return load(filepath)

        if len(identifier_and_default) == 2:
            return identifier_and_default[1]
        else:
            raise KeyError(f"identifier '{identifier}' does not exist in {self}")

    def __getitem__(self, identifier):
        return self.get(identifier)

    def __contains__(self, identifier):
        try:
            self.get(identifier)
        except (KeyError, FileNotFoundError):
            return False
        else:
            return True

    def gets(self, *identifiers) -> Any:
        for identifier in identifiers:
            yield self.get(identifier)

    def remove(self, identifier):
        file = self._find_file(self, identifier)
        filepath: Path = self / file
        assert filepath.exists()
        if filepath.is_file():
            os.remove(filepath)
        else:
            shutil.rmtree(filepath)

    def put(self, value, identifier, dtype=None):
        if self._rank > 0:
            return

        if not self.exists():
            self.mkdir(exist_ok=True, parents=True)
        assert self.is_dir()

        dtype = dtype or self._default_dtype
        if dtype is None:
            save(self / identifier, value)
        else:
            save(self / identifier, as_type(value, dtype))

        # post examination to prevent duplicated file stems
        assert self._find_file(self, identifier) is not None

    def __setitem__(self, identifier, value):
        self.put(value, identifier)

    def __delitem__(self, key):
        self.remove(key)

    def dirs(self):
        return list(self._dirs())

    def _dirs(self):
        if not self.is_dir():
            return []
        for filename in os.listdir(self):
            filepath = self / filename
            if not self._is_hidden(filepath):
                if filepath.is_dir():
                    yield Path(filepath).stem

    def files(self):
        return list(self._files())

    def _files(self):
        if not self.is_dir():
            return []
        for filename in os.listdir(self):
            filepath = self / filename
            if not self._is_hidden(filepath):
                if filepath.is_file():
                    yield Path(filepath).stem

    @staticmethod
    def _is_hidden(filepath):
        parts = Path(filepath).parts
        for part in parts:
            if str(part).startswith("."):
                return True
        return False

    def put_cfg(self, method: MethodType, cfg: dict):
        assert len(method.__qualname__) > 0
        (self / "_cfg").put(cfg, method.__qualname__, "yaml")

    def get_cfg(self, method: MethodType):
        assert len(method.__qualname__) > 0
        return (self / "_cfg").get(method.__qualname__, None)

    def get_log_file(self, method: MethodType):
        assert len(method.__qualname__) > 0
        log_dir = self / "_log"
        log_dir.mkdir(exist_ok=True, parents=True)
        return log_dir / f"{method.__qualname__}.log"

    @property
    def environ(self) -> Workspace:
        return self / "_environ"

    def __iter__(self):
        return iter(self.files())

    def __len__(self):
        return len(self.files())

    def _keytransform(self, key):
        return key

    def link_from(self, link_key_name: str, src_workspace: str, src_key: str):
        src_workspace = __class__(src_workspace)
        src_path: Path = src_workspace.get_filepath(src_key)
        ext = src_path.suffix
        link_file = self / f"{link_key_name}{ext}"
        if link_file.is_symlink():
            link_file.unlink()
        link_file.symlink_to(src_path.resolve())


class Checkpoint(Workspace):
    def register_item(self, obj: Any, identifier: str, dtype: str):
        if not hasattr(self, "_items"):
            self._items = {}
        self._items[identifier] = as_type(obj, dtype)

    def save_to(self, *stems):
        folder = __class__(os.path.join(self, *stems))
        for key, typed_obj in self._items.items():
            folder.put(typed_obj, key)
