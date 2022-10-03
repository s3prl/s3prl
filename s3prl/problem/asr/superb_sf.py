"""
The setting of Superb SF

Authors
  * Yung-Sung Chuang 2021
  * Heng-Jui Chang 2022
  * Shu-wen Yang 2022
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import MISSING

from s3prl.dataio.corpus.snips import SNIPS
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler
from s3prl.util.download import _urls_to_filepaths

from .superb_asr import SuperbASR

VOCAB_URL = "https://huggingface.co/datasets/s3prl/SNIPS/raw/main/character.txt"
SLOTS_URL = "https://huggingface.co/datasets/s3prl/SNIPS/raw/main/slots.txt"

__all__ = [
    "audio_snips_for_slot_filling",
    "SuperbSF",
]


def audio_snips_for_slot_filling(
    target_dir: str,
    cache_dir: str,
    dataset_root: str,
    train_speakers: List[str],
    valid_speakers: List[str],
    test_speakers: List[str],
    get_path_only: bool = False,
):
    target_dir = Path(target_dir)

    train_path = target_dir / f"train.csv"
    valid_path = target_dir / f"valid.csv"
    test_paths = [target_dir / f"test.csv"]

    if get_path_only:
        return train_path, valid_path, test_paths

    corpus = SNIPS(dataset_root, train_speakers, valid_speakers, test_speakers)
    train_data, valid_data, test_data = corpus.data_split

    def dict_to_csv(data_dict, csv_path):
        keys = sorted(list(data_dict.keys()))
        fields = sorted(data_dict[keys[0]].keys())
        data = dict()
        for field in fields:
            data[field] = []
            for key in keys:
                data[field].append(data_dict[key][field])
        data["id"] = keys
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    dict_to_csv(train_data, train_path)
    dict_to_csv(valid_data, valid_path)
    dict_to_csv(test_data, test_paths[0])

    return train_path, valid_path, test_paths


class SuperbSF(SuperbASR):
    def default_config(self) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=None,
            remove_all_cache=False,
            prepare_data=dict(
                dataset_root=MISSING,
                train_speakers=[
                    "Ivy",
                    "Joanna",
                    "Joey",
                    "Justin",
                    "Kendra",
                    "Kimberly",
                    "Matthew",
                    "Salli",
                ],
                valid_speakers=["Aditi", "Amy", "Geraint", "Nicole"],
                test_speakers=["Brian", "Emma", "Raveena", "Russell"],
            ),
            prepare_tokenizer_data=dict(),
            build_tokenizer=dict(
                tokenizer_name=None,
                vocab_type="character",
                vocab_file=VOCAB_URL,
                slots_file=SLOTS_URL,
            ),
            build_dataset=dict(),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=32,
                    shuffle=True,
                ),
                valid=dict(
                    batch_size=1,
                ),
                test=dict(
                    batch_size=1,
                ),
            ),
            build_upstream=dict(
                name="fbank",
            ),
            build_featurizer=dict(
                layer_selections=None,
                normalize=False,
            ),
            build_downstream=dict(
                model_conf=dict(
                    module="LSTM",
                    proj_size=1024,
                    hidden_size=[1024, 1024],
                    dropout=[0.2, 0.2],
                    layer_norm=[False, False],
                    proj=[False, False],
                    sample_rate=[1, 1],
                    sample_style="concat",
                    bidirectional=True,
                ),
                specaug_conf=dict(
                    freq_mask_width_range=(0, 50),
                    num_freq_mask=4,
                    time_mask_width_range=(0, 40),
                    num_time_mask=2,
                ),
            ),
            build_model=dict(
                upstream_trainable=False,
            ),
            build_task=dict(
                log_metrics=[
                    "wer",
                    "cer",
                    "slot_type_f1",
                    "slot_value_cer",
                    "slot_value_wer",
                    "slot_edit_f1_full",
                    "slot_edit_f1_part",
                ],
            ),
            build_optimizer=dict(
                name="Adam",
                conf=dict(
                    lr=1.0e-4,
                ),
            ),
            build_scheduler=dict(
                name="ExponentialLR",
                gamma=0.9,
            ),
            save_model=dict(),
            save_task=dict(),
            train=dict(
                total_steps=200000,
                log_step=100,
                eval_step=2000,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate=1,
                valid_metric="slot_type_f1",
                valid_higher_better=True,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
        )

    def prepare_data(
        self,
        prepare_data: dict,
        target_dir: str,
        cache_dir: str,
        get_path_only: bool = False,
    ):
        """
        Prepare the task-specific data metadata (path, labels...).
        By default call :obj:`audio_snips_for_slot_filling` with :code:`**prepare_data`

        Args:
            prepare_data (dict): same in :obj:`default_config`, support arguments in :obj:`audio_snips_for_slot_filling`
            target_dir (str): Parse your corpus and save the csv file into this directory
            cache_dir (str): If the parsing or preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            get_path_only (str): Directly return the filepaths no matter they exist or not.

        Returns:
            tuple

            1. train_path (str)
            2. valid_path (str)
            3. test_paths (List[str])

            Each path (str) should be a csv file containing the following columns:

            ====================  ====================
            column                description
            ====================  ====================
            id                    (str) - the unique id for this data point
            wav_path              (str) - the absolute path of the waveform file
            transcription         (str) - a text string
            ====================  ====================
        """
        return audio_snips_for_slot_filling(
            **self._get_current_arguments(flatten_dict="prepare_data")
        )

    def build_batch_sampler(
        self,
        build_batch_sampler: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        dataset,
    ):
        """
        Return the batch sampler for torch DataLoader.

        Args:
            build_batch_sampler (dict): same in :obj:`default_config`

                ====================  ====================
                key                   description
                ====================  ====================
                train                 (dict) - arguments for :obj:`FixedBatchSizeBatchSampler`
                valid                 (dict) - arguments for :obj:`FixedBatchSizeBatchSampler`
                test                  (dict) - arguments for :obj:`FixedBatchSizeBatchSampler`
                ====================  ====================

            target_dir (str): Current experiment directory
            cache_dir (str): If the preprocessing takes too long time, save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            mode (str): train/valid/test
            data_csv (str): the :code:`mode` specific csv from :obj:`prepare_data`
            dataset: the dataset from :obj:`build_dataset`

        Returns:
            batch sampler for torch DataLoader
        """

        @dataclass
        class Config:
            train: dict = None
            valid: dict = None
            test: dict = None

        conf = Config(**build_batch_sampler)

        if mode == "train":
            return FixedBatchSizeBatchSampler(dataset, **(conf.train or {}))
        elif mode == "valid":
            return FixedBatchSizeBatchSampler(dataset, **(conf.valid or {}))
        elif mode == "test":
            return FixedBatchSizeBatchSampler(dataset, **(conf.test or {}))
        else:
            raise ValueError(f"Unsupported mode: {mode}")
