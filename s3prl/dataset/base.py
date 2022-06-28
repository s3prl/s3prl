from __future__ import annotations

import abc
import logging
import pickle
from copy import deepcopy
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

from s3prl import Container, Object, cache
from s3prl.util import registry

logger = logging.getLogger(__name__)


class AugmentedDynamicItemDataset(DynamicItemDataset):
    def __init__(
        self,
        data,
        dynamic_items=[],
        output_keys=[],
        tools: dict = {},
    ):
        super().__init__(data, dynamic_items, output_keys)
        self._tools = {}
        for name, item in tools.items():
            self.add_tool(name, item)

    def __init_subclass__(cls) -> None:
        registry.put(cls.__name__)(cls)
        return super().__init_subclass__()

    def _dynamic_tools(self, id, name):
        return self._tools[name]

    def add_tool(self, name: str, item: Any) -> None:
        """
        Store the :code:`item` in this dataset with the name :code:`name` so it can be used in
        :code:`__getitem__`. That is, you can retrieve the :code:`item` with the :code:`takes` argument
        of :obj:`add_dynamic_item`.

        .. code-block:: python

            def tokenize_func(tokenizer, text):
                return tokenizer(text)

            self.add_tool("tokenizer", tokenizer)
            self.add_dynamic_item(tokenize_func, takes=["tokenizer", "text"], provides="tokenized_ids")

        You can also later retreive this tool by :obj:`get_tool` or :obj:`all_tools`
        """
        self._tools[name] = item
        self.add_dynamic_item(
            partial(self._dynamic_tools, name=name), takes="id", provides=name
        )

    def add_tools(self, tools: dict) -> None:
        """
        Store each key-value pair in :code:`tools` as a tool. See :obj:`add_tool` for more information
        """
        for key, value in tools.items():
            self.add_tool(key, value)

    def get_tool(self, key) -> Any:
        """
        See :obj:`add_tool` for more information
        """
        return self._tools[key]

    def has_tool(self, key) -> bool:
        """
        Checks whether has a tool named :code:`key`.
        """
        return key in self._tools

    def all_tools(self, copy=True) -> dict:
        """
        Return:
            dict

            Containing all the tools in :code:`name: value` pairs.
            See :obj:`add_tool` for more information
        """
        return deepcopy(self._tools) if copy else self._tools

    def update_output_keys(self, keys: dict) -> None:
        """
        Compared to :obj:`set_output_keys`, this method update the output keys mapping
        instead of replace it with a new dictionary. This can be useful when you only
        want to replace a few mapping and leave others unchanged.
        """
        mapping = self.pipeline.output_mapping.copy()
        mapping.update(keys)
        self.set_output_keys(mapping)

    def replace_output_key(self, old_key: str, new_key: str):
        """
        Given an existing mapping:

        .. code-block:: python

            {
                "x": "wav",
                "x_len": "wav_len",
            }

        If :code:`old_key = "wav"` and :code:`new_key = "noisy_wav"`, after calling this
        method the output mapping will become

        .. code-block:: python

            {
                "x": "noisy_wav",
                "x_len": "wav_len",
            }

        This can be useful when you are augmenting the original output.
        """
        mapping = self.pipeline.output_mapping.copy()
        for key in list(mapping.keys()):
            value = mapping[key]
            if value == old_key:
                mapping[key] = new_key
                self.update_output_keys(mapping)

    def keys(self) -> List[str]:
        """
        List all the :code:`static_item` and :code:`dynamic_item` in the dataset.
        :code:`static_item` resides directly in the memory and are given by the dataset
        initialization dictionary. :code:`dynamic_item` are content computed
        on-the-fly basing on :code:`static_item`.
        """
        available_keys: List[str] = list(self.pipeline.key_to_node.keys())
        for dynamic_item in self.pipeline.dynamic_items:
            provides = dynamic_item.provides
            assert isinstance(provides, (list, tuple))
            available_keys += provides
        available_keys = [
            key
            for key in available_keys
            if not key.startswith("_") and key not in self._tools
        ]
        return available_keys

    def __getitem__(self, index):
        """
        This remain all the usage of the original SpeechBrain DynamicItemDataset.__getitem__,
        except that by default it uses :obj:`keys` as the default :code:`output_keys`
        """
        if len(self.pipeline.output_mapping) == 0:
            with self.output_keys_as(self.keys()):
                return super().__getitem__(index)
        else:
            return super().__getitem__(index)

    def save_checkpoint(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_checkpoint(cls, path):
        with open(path, "rb") as file:
            obj = pickle.load(file)
        return obj


class DataPipe(object):
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
        *pipes_or_cfgs: List[Union[DataPipe, Type, dict]],
        **global_cfg,
    ) -> None:
        assert len(pipes_or_cfgs) > 0
        if isinstance(pipes_or_cfgs[0], DataPipe):
            pipes, pipe_cfgs = pipes_or_cfgs, None
        elif isinstance(pipes_or_cfgs[0], dict):
            pipes, pipe_cfgs = None, pipes_or_cfgs
        else:
            raise ValueError

        if pipes is None:
            pipes = []
            for pipe_cfg in pipe_cfgs:
                pipe_cfg = Container(pipe_cfg)
                pipe = pipe_cfg._cls(**pipe_cfg.kwds(), **global_cfg)
                pipes.append(pipe)

        self._pipes = pipes

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        for pipe in self._pipes:
            dataset = pipe(dataset)
        return dataset


def default_collate_fn(samples, padding_value: int = 0):
    """
    Each item in **DynamicItemDataset** is a :obj:`s3prl.base.container.Container`
    This function pad (or transform into numpy list) a batch of :obj:`s3prl.base.container.Container`.

    Args:
        samples (List[:obj:`s3prl.base.container.Container`]): Suppose each Container is in

            .. code-block:: yaml

                wav: a single waveform
                label: a single string

    Return:
        :obj:`s3prl.base.container.Container`

        .. code-block:: yaml

            wav: padded waveforms
            label: np.array([a list of string labels])
    """
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
    return Container(padded_samples)


class DataLoader(data.DataLoader):
    """
    This DataLoader is just a helper class to enforce the use of the **batch_sampler**.
    Since S3PRL only provides **batch_sampler**. See :obj:`s3prl.sampler`
    """

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
