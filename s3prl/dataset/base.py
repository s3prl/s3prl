from __future__ import annotations

import abc
import logging
import pickle
from copy import deepcopy
from dataclasses import dataclass, fields
from enum import Enum
from functools import partial
from inspect import isclass, isfunction, ismethod
from typing import Any, List, Type, Union

import numpy as np
import torch
import torchaudio
from joblib import Parallel, delayed
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_pipeline import DynamicItem
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from tqdm import tqdm

from s3prl import Object, Output, cache
from s3prl.base.container import _qualname_to_cls
from s3prl.util import registry

logger = logging.getLogger(__name__)


class AugmentedDynamicItemDataset(DynamicItemDataset):
    def __init__(
        self,
        data,
        dynamic_items=[],
        output_keys=[],
        tools: dict = {},
        global_stats: dict = {},
    ):
        super().__init__(data, dynamic_items, output_keys)
        self._tools = {}
        for name, item in tools.items():
            self.add_tool(name, item)

        self._global_stats = {}
        for name, item in global_stats.items():
            self.add_global_stats(name, item)

    def __init_subclass__(cls) -> None:
        registry.put(cls.__name__)(cls)
        return super().__init_subclass__()

    def _dynamic_global_stats(self, id, name):
        return self._global_stats[name]

    def add_global_stats(self, name: str, item: Any):
        self._global_stats[name] = item
        self.add_dynamic_item(
            partial(self._dynamic_global_stats, name=name), takes="id", provides=name
        )

    def _dynamic_tools(self, id, name):
        return self._tools[name]

    def add_tool(self, name: str, item: Any):
        """
        Store the "item" in this dataset with the name "name" so it can be used in
        __getitem__. That is, you can retrieve the "item" with the "takes" argument
        of self.add_dynamic_item.

        E.g.
            def tokenize_func(tokenizer, text):
                return tokenizer(text)

            self.add_tool("tokenizer", tokenizer)
            self.add_dynamic_item(tokenize_func, takes=["tokenizer", "text"], provides="tokenized_ids")

        You can also later retreive this tool by self.get_tool or self.all_tools
        """
        self._tools[name] = item
        self.add_dynamic_item(
            partial(self._dynamic_tools, name=name), takes="id", provides=name
        )

    def add_tools(self, tools: dict):
        """
        Store each key-value pair in "tools" as a tool. See self.add_tool for more information
        """
        for key, value in tools.items():
            self.add_tool(key, value)

    def get_tool(self, key):
        """
        See self.add_tool
        """
        return self._tools[key]

    def has_tool(self, key):
        """
        Checks whether has a tool named `key`.
        """
        return key in self._tools

    def all_tools(self, copy=True):
        """
        See self.add_tool
        """
        return deepcopy(self._tools) if copy else self._tools

    def add_output_keys(self, keys):
        if isinstance(keys, list):
            keys = {key: key for key in keys}
        mapping = self.pipeline.output_mapping.copy()
        mapping.update(keys)
        self.set_output_keys(mapping)

    def add_dynamic_item(self, func, takes=None, provides=None):
        if isinstance(func, DynamicItem):
            logger.warning(f"Ignoring default takes: {takes}, and provides {provides}")
            takes = func.takes
            provides = func.provides
        super().add_dynamic_item(func, takes, provides)

    def add_dynamic_item_and_metadata(self, func, takes=None, provide=None):
        """
        The function should take `metadata` as an optional argument
        The output dict will auto add a key: f"{provide}_metadata"
        """

        if isinstance(func, DynamicItem):
            logger.warning(f"Ignoring default takes: {takes}, and provide {provide}")
            takes = func.takes
            provide = func.provides
            func = func.func
        assert isfunction(func) or ismethod(func)

        if not isinstance(provide, str):
            assert len(provide) == 1
            provide = provide[0]
        else:
            provide = provide

        self.add_dynamic_item(
            partial(func, metadata=False), takes=takes, provides=provide
        )

        if isinstance(takes, str):
            takes = [takes]

        with self.output_keys_as(["id"] + takes):

            def get_item(item):
                fields = [item[take] for take in takes]
                return fields

            id_to_take_items = {item["id"]: get_item(item) for item in self}

        ids = sorted(id_to_take_items.keys())
        all_take_items = [id_to_take_items[idx] for idx in ids]
        metadatas = self._precompute_metadatas(func, all_take_items)
        id_to_metadata = {idx: metadata for idx, metadata in zip(ids, metadatas)}

        mapping_name = f"_id_to_metadata_for_{provide}"
        self.add_global_stats(mapping_name, id_to_metadata)
        self.add_dynamic_item(
            self._get_metadata,
            takes=["id", mapping_name],
            provides=f"{provide}_metadata",
        )

    @staticmethod
    def _get_metadata(id, mapping):
        return mapping[id]

    @staticmethod
    @cache(signatures=["func", "all_take_items"])
    def _precompute_metadatas(func, all_take_items, n_jobs: int = 8):
        metadatas = Parallel(n_jobs=n_jobs)(
            delayed(partial(func, metadata=True))(*take_items)
            for take_items in tqdm(all_take_items, desc="precompute metadata")
        )
        return metadatas

    @property
    def available_keys(self):
        available_keys: List[str] = list(self.pipeline.key_to_node.keys())
        for dynamic_item in self.pipeline.dynamic_items:
            provides = dynamic_item.provides
            assert isinstance(provides, (list, tuple))
            available_keys += provides
        available_keys = [
            key
            for key in available_keys
            if not key.startswith("_")
            and key not in self._global_stats
            and key not in self._tools
        ]
        return available_keys

    def __getitem__(self, index):
        if len(self.pipeline.output_mapping) == 0:
            with self.output_keys_as(self.available_keys):
                return super().__getitem__(index)
        else:
            return super().__getitem__(index)

    def save_checkpoint(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def keys(self):
        return list(self.data.keys())

    @classmethod
    def load_checkpoint(cls, path):
        with open(path, "rb") as file:
            obj = pickle.load(file)
        return obj


@dataclass
class DatasetBuilder:
    audio_sample_rate: int = 16000

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, DynamicItem):
            value.func = value.func.__get__(self)
        return value

    def load_audio(self, wav_path, metadata: bool = False):
        if not metadata:
            torchaudio.set_audio_backend("sox_io")
            wav, sr = torchaudio.load(wav_path)
            if sr != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
                wav = resampler(wav)
            wav = wav.view(-1, 1)
            return wav
        else:
            torchaudio.set_audio_backend("sox_io")
            info = torchaudio.info(wav_path)
            ratio = self.audio_sample_rate / info.sample_rate
            return dict(
                sample_rate=self.audio_sample_rate,
                num_frames=round(info.num_frames * ratio),
                num_channels=1,
            )


@dataclass
class DataPipe:
    n_jobs: int = 6

    def __call__(
        self, dataset: Union[dict, AugmentedDynamicItemDataset], **kwds
    ) -> Any:
        if isinstance(dataset, dict):
            dataset = AugmentedDynamicItemDataset(dataset)
        dataset.add_tools(kwds)
        return self.forward(dataset)

    @abc.abstractmethod
    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        raise NotImplementedError

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, DynamicItem):
            value.func = value.func.__get__(self)
        return value

    def __init_subclass__(cls) -> None:
        registry.put(cls.__name__)(cls)
        return super().__init_subclass__()


class SequentialDataPipe(DataPipe):
    def __init__(
        self,
        *pipes_or_classes: List[Union[DataPipe, Type]],
        config: dict = None,
        configs: List[dict] = None,
    ) -> None:
        assert len(pipes_or_classes) > 0
        if isinstance(pipes_or_classes[0], DataPipe):
            pipes, pipe_classes = pipes_or_classes, None
        elif isclass(pipes_or_classes[0]):
            pipes, pipe_classes = None, pipes_or_classes
        else:
            raise ValueError

        if pipes is None:
            assert int(configs is not None) + int(config is not None) == 1
        if pipes is not None:
            assert configs is None and config is None

        if pipes is None:
            if config is not None:
                configs = [config for _ in pipe_classes]
            assert len(configs) == len(pipe_classes)

            pipes = []
            for pipe_class, config in zip(pipe_classes, configs):
                related_fields = [field.name for field in fields(pipe_class)]
                related_config = {
                    k: v for k, v in config.items() if k in related_fields
                }
                pipes.append(pipe_class(**related_config))

        self._pipes = pipes

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        for pipe in self._pipes:
            dataset = pipe(dataset)
        return dataset


def default_collate_fn(samples, padding_value: int = 0):
    assert isinstance(samples[0], dict)
    keys = samples[0].keys()
    padded_samples = dict()
    for key in keys:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], (int, float)):
            values = torch.tensor(values)
        elif isinstance(values[0], np.ndarray):
            values = [torch.from_numpy(value) for value in values]
            values = pad_sequence(values, batch_first=True, padding_value=padding_value)
        elif isinstance(values[0], torch.Tensor):
            values = pad_sequence(values, batch_first=True, padding_value=padding_value)
        else:
            values = np.array(values, dtype="object")
        padded_samples[key] = values
    return Output(padded_samples)


class DataLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_sampler,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        collate_fn = collate_fn or default_collate_fn

        super().__init__(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )


class Mode(Enum):
    FULL = 0
    METADATA = 1


_mode = Mode.FULL


def set_mode(mode: str):
    global _mode
    if isinstance(mode, Mode):
        _mode = mode
    elif mode == "FULL":
        _mode = Mode.FULL
    elif mode == "METADATA":
        _mode = Mode.METADATA


def in_metadata_mode():
    return _mode == Mode.METADATA


class metadata_mode:
    def __init__(self):
        self.prev = None

    def __enter__(self):
        self.prev = _mode
        set_mode(Mode.METADATA)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        set_mode(self.prev)


class Dataset(Object, data.Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def in_metadata_mode():
        return in_metadata_mode()

    @abc.abstractmethod
    def collate_fn(self, samples):
        pass
