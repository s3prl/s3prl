# Copyright 2022 ByteDance Inc.
# CoBERT: Self-Supervised Speech Representation Learning Through Code Representation Learning (https://arxiv.org/abs/2210.04062)
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq (https://github.com/facebookresearch/fairseq)

import logging
from typing import List, Union, Optional, Any

from .audio_code_dataset import AudioCodeDataset

logger = logging.getLogger(__name__)


class CobertDataset(AudioCodeDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            pad_list: List[str],
            eos_list: List[str],
            fine_tuning: bool = False,
            label_processors: Optional[List[Any]] = None,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            code_input: bool = False,
            code_paths: List[str] = None,
            code_processors: Optional[List[Any]] = None,
            code_rate: Optional[float] = None,
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            label_paths=label_paths,
            label_rates=label_rates,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=label_processors,
            max_keep_sample_size=max_keep_sample_size,
            min_keep_sample_size=min_keep_sample_size,
            max_sample_size=max_sample_size,
            shuffle=shuffle,
            pad_audio=pad_audio,
            normalize=normalize,
            store_labels=store_labels,
            random_crop=random_crop,
            single_target=single_target,
            code_input=code_input,
            code_paths=code_paths,
            code_processors=code_processors,
            code_rate=code_rate
        )
        self.fine_tuning = fine_tuning

    def collater(self, samples):
        batch = super().collater(samples)
        assert self.single_target, "Not supported multiple single target yet."
        if not self.fine_tuning:
            if "target" not in batch:
                return batch
            # rename to fit the real net input
            batch["net_input"]["source_codes"] = batch.pop("target")
        else:
            batch = batch
        return batch
