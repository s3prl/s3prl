"""
The setting of Superb SF

Authors
  * Yung-Sung Chuang 2021
  * Heng-Jui Chang 2022
  * Leo 2022
"""

import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import MISSING

from s3prl.dataio.corpus.snips import SNIPS
from s3prl.dataio.dataset import EncodeText, LoadAudio, get_info
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler, SortedSliceSampler

from .superb_asr import SuperbASR, prepare_common_tokenizer

# Mapping for character-slot tokenizer (SNIPS)
translator = str.maketrans('ÁÃÄÅÆÇÈÉÊËÍÏÐÒÓÔÖØÚÛĘŃŌŞŪ"', "AAAAACEEEEIIDOOOOOUUENOSU ")

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
        data_ids = sorted(list(data_dict.keys()))
        fields = sorted(data_dict[data_ids[0]].keys())
        data = defaultdict(list)
        for data_id in data_ids:
            data_point = data_dict[data_id]

            trans = data_point["transcription"]
            trans = trans.replace("楽園追放", "EXPELLED")
            trans = trans.replace("官方杂志", "")
            trans = trans.replace("–", "-")
            trans = trans.replace("&", " AND ")
            trans = trans.translate(translator)
            trans = re.sub(" +", " ", trans).strip(" ")

            words = trans.split(" ")
            iobs = data_point["iob"].split(" ")
            assert len(words) == len(iobs)

            filtered_words = []
            filtered_iobs = []
            for word, iob in zip(words, iobs):
                if word in "?!.,;-–…":
                    continue
                filtered_words.append(word)
                filtered_iobs.append(iob)

            assert len(filtered_words) == len(filtered_iobs)
            data_point["transcription"] = " ".join(filtered_words)
            data_point["iob"] = " ".join(filtered_iobs)

            for field in fields:
                data[field].append(data_point[field])

        data["id"] = data_ids
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
                vocab_type="character",
            ),
            build_dataset=dict(),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=32,
                    max_length=300000,
                ),
                valid=dict(
                    batch_size=1,
                ),
                test=dict(
                    batch_size=1,
                ),
            ),
            build_upstream=dict(
                name=MISSING,
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
            transcription         (str) - a text string where words are separted by a space.
                                    Eg. "I want to fly from Taipei to New York"
            iob                   (str) - iob tags, use "O" if no tag, every word should have a tag, separted by a space.
                                    Eg. "O O O O O from_location O to_location to_location"
            ====================  ====================
        """
        return audio_snips_for_slot_filling(
            **self._get_current_arguments(flatten_dict="prepare_data")
        )

    def prepare_tokenizer_data(
        self,
        prepare_tokenizer_data: dict,
        target_dir: str,
        cache_dir: str,
        train_csv: str,
        valid_csv: str,
        test_csvs: str,
        get_path_only: bool = False,
    ):
        data_dir = target_dir / "tokenizer_data"
        if get_path_only:
            return data_dir

        train_df = pd.read_csv(train_csv)
        valid_df = pd.read_csv(valid_csv)
        test_dfs = [pd.read_csv(test_csv) for test_csv in test_csvs]
        iob_lines = pd.concat([train_df, valid_df, *test_dfs], axis=0)["iob"].tolist()
        iobs = []
        for line in iob_lines:
            iobs.extend(line.split(" "))
        iobs = list(sorted(set(iobs)))

        Path(data_dir).mkdir(parents=True, exist_ok=True)

        with open(data_dir / "slot.txt", "w") as f:
            f.writelines([f"{iob}\n" for iob in iobs])

        train_df = pd.read_csv(train_csv)
        texts = train_df["transcription"].tolist()

        with open(data_dir / "text.txt", "w") as f:
            f.writelines([f"{t}\n" for t in texts])

        return data_dir

    def build_tokenizer(
        self,
        build_tokenizer: dict,
        target_dir: str,
        cache_dir: str,
        tokenizer_data_path: str,
        get_path_only: bool = False,
    ):
        return prepare_common_tokenizer(
            target_dir,
            cache_dir,
            Path(tokenizer_data_path) / "text.txt",
            get_path_only,
            None,
            None,
            slots_file=Path(tokenizer_data_path) / "slot.txt",
            **build_tokenizer,
        )

    def build_dataset(
        self,
        build_dataset: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        tokenizer_path: str,
    ):
        csv = pd.read_csv(data_csv)

        audio_loader = LoadAudio(csv["wav_path"].tolist())

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        text_encoder = EncodeText(
            csv["transcription"].tolist(), tokenizer, iob=csv["iob"].tolist()
        )
        ids = csv["id"].tolist()

        class SlotFillingDataset:
            def __len__(self):
                return len(audio_loader)

            def __getitem__(self, index: int):
                audio = audio_loader[index]
                text = text_encoder[index]
                return {
                    "x": audio["wav"],
                    "x_len": audio["wav_len"],
                    "class_ids": text["class_ids"],
                    "labels": text["labels"],
                    "unique_name": ids[index],
                }

        dataset = SlotFillingDataset()
        return dataset

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
                train                 (dict) - arguments for :obj:`SortedSliceSampler`
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
            wav_lens = get_info(
                dataset, "x_len", cache_dir=Path(target_dir) / "train_stats"
            )
            sampler = SortedSliceSampler(wav_lens, **(conf.train or {}))
            return sampler
        elif mode == "valid":
            return FixedBatchSizeBatchSampler(dataset, **(conf.valid or {}))
        elif mode == "test":
            return FixedBatchSizeBatchSampler(dataset, **(conf.test or {}))
        else:
            raise ValueError(f"Unsupported mode: {mode}")
