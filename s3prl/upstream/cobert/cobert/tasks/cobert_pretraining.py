# Copyright 2022 ByteDance Inc.
# CoBERT: Self-Supervised Speech Representation Learning Through Code Representation Learning (https://arxiv.org/abs/2210.04062)
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq (https://github.com/facebookresearch/fairseq)

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List

from omegaconf import II, MISSING, OmegaConf

from fairseq.data import Dictionary
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import register_task, FairseqTask

from ..data.cobert_dataset import CobertDataset


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )


@dataclass
class CobertPretrainingConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={"help": "path to data directory."
                          "contains split.txt and dict.txt for code;"
                          "contains split.tsv for audio."}
    )
    fine_tuning: bool = field(
        default=False, metadata={"help": "whether in fine-tuning mode."}
    )
    label: Optional[str] = field(
        default="km",
        metadata={"help": "extension of the label file to load, used for code file."}
    )
    code_rate: float = field(
        default=50,
        metadata={"help": "code frame rate. -1.0 for sequence label"},
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={"help": "path to data directory used in fine-tuning."
                          "contains split.txt and dict.txt for ltr;"}
    )
    label_suffix: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load. used for fine-tuning"}
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "code frame rate. -1.0 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "audio sample rate. audio files will be up/down "
                    "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes audio to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    random_crop: bool = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: bool = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )


@register_task("cobert_pretraining", dataclass=CobertPretrainingConfig)
class CobertPretrainingTask(FairseqTask):
    """
    This task is responsible for code input tasks.
    If pre-training, then code is the input. No explicit output is provided.
    If fine-tuning, then code is the input, and ltr is the output.
    """
    def __init__(self, cfg: CobertPretrainingConfig, code_dict, ltr_dict=None):
        super().__init__(cfg)
        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning
        logger.info(f"Is in fine_tuning mode? {self.fine_tuning}")

        self.code_dictionary = code_dict
        self.letter_dictionary = ltr_dict

    @classmethod
    def setup_task(cls, cfg: CobertPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (CobertPretrainingConfig): configuration of this task
        """
        code_dict_path = os.path.join(cfg.data, f"dict.{cfg.label}.txt")
        if not os.path.exists(code_dict_path):
            code_dict = None
            logger.warning(f"Cannot load dict from {code_dict_path}. Make sure it is the expected behavior.")
        else:
            logger.info(f"Loaded dict from {code_dict_path}.")
            code_dict = cls.load_dictionary(code_dict_path)

        ltr_dict = None
        if cfg.fine_tuning:
            ltr_dict_path = os.path.join(cfg.label_dir, f"dict.{cfg.label_suffix}.txt")
            if not os.path.exists(ltr_dict_path):
                ltr_dict = None
                logger.warning(f"Cannot load dict from {ltr_dict_path}. Make sure it is the expected behavior.")
            else:
                logger.info(f"Loaded dict from {ltr_dict_path}.")
                ltr_dict = cls.load_dictionary(ltr_dict_path)
        return cls(cfg, code_dict, ltr_dict)

    def load_dataset(
            self,
            split: str,
            combine: bool = False,
            task_cfg: FairseqDataclass = None,
            **kwargs,
    ):
        # load audio and corresponding code dataset
        # predefined files to be loaded
        audio_manifest = f"{self.cfg.data}/{split}.tsv"
        code_file = os.path.join(self.cfg.data, f"{split}.{self.cfg.label}")
        if not self.fine_tuning:
            # the dataset only returns speech and code.
            # speech as the 'source', code as the 'source_codes'.
            # but only the code is expected to be used.
            self.datasets[split] = CobertDataset(
                audio_manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=[code_file],
                label_rates=[self.cfg.code_rate],
                pad_list=[self.source_dictionary.pad()],
                eos_list=[self.source_dictionary.eos()],
                fine_tuning=False,
                label_processors=[LabelEncoder(self.source_dictionary)],
                max_keep_sample_size=self.cfg.max_keep_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                shuffle=True,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=True,
                random_crop=self.cfg.random_crop,
                # TODO: support only ONE kind of code now.
                single_target=True,
                code_input=False,
                code_paths=None,
                code_processors=None,
                code_rate=None
            )
        else:
            label_file = os.path.join(self.cfg.label_dir, f"{split}.{self.cfg.label_suffix}")
            # the dataset returns the code, and label.
            # code as the 'source', label as the 'target'
            self.datasets[split] = CobertDataset(
                audio_manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=[label_file],
                label_rates=[self.cfg.label_rate],
                pad_list=[self.target_dictionary.pad()],
                eos_list=[self.target_dictionary.eos()],
                fine_tuning=True,
                label_processors=[LabelEncoder(self.target_dictionary)],
                max_keep_sample_size=self.cfg.max_keep_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                shuffle=True,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=True,
                random_crop=self.cfg.random_crop,
                # TODO: support only ONE kind of code now.
                single_target=True,
                code_input=True,
                code_paths=[code_file],
                code_processors=[LabelEncoder(self.source_dictionary)],
                code_rate=self.cfg.code_rate
            )

    @property
    def source_dictionary(self) -> Dictionary:
        return self.code_dictionary

    @property
    def target_dictionary(self):
        return self.letter_dictionary
