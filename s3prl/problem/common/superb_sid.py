"""
The setting of Superb SID

Authors
  * Po-Han Chi 2021
  * Leo 2022
"""

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import MISSING

from s3prl.dataio.corpus.voxceleb1sid import VoxCeleb1SID
from s3prl.dataio.dataset import EncodeCategory, LoadAudio
from s3prl.dataio.encoder.category import CategoryEncoder
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler
from s3prl.nn.linear import MeanPoolingLinear

from .run import Common

logger = logging.getLogger(__name__)


__all__ = [
    "voxceleb1_for_sid",
    "SuperbSID",
]


def voxceleb1_for_sid(
    target_dir: str,
    cache_dir: str,
    dataset_root: str,
    n_jobs: int = 6,
    get_path_only: bool = False,
):
    """
    Prepare VoxCeleb1 for SID following :obj:`SuperbSID.prepare_data` format.

    Args:
        dataset_root (str): The root path of VoxCeleb1
        n_jobs (int): to speed up the corpus parsing procedure
        **others: refer to :obj:`SuperbSID.prepare_data`
    """
    target_dir = Path(target_dir)

    train_path = target_dir / f"train.csv"
    valid_path = target_dir / f"valid.csv"
    test_paths = [target_dir / f"test.csv"]

    if get_path_only:
        return train_path, valid_path, test_paths

    corpus = VoxCeleb1SID(dataset_root, n_jobs)
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


class SuperbSID(Common):
    """
    The standard SUPERB SID task
    """

    def default_config(self) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=None,
            remove_all_cache=False,
            prepare_data=dict(
                dataset_root=MISSING,
            ),
            build_encoder=dict(),
            build_dataset=dict(
                train=dict(
                    max_secs=8.0,
                ),
            ),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=8,
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
                name=MISSING,
            ),
            build_featurizer=dict(
                layer_selections=None,
                normalize=False,
            ),
            build_downstream=dict(
                hidden_size=256,
            ),
            build_model=dict(
                upstream_trainable=False,
            ),
            build_task=dict(),
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
                log_step=500,
                eval_step=5000,
                save_step=1000,
                gradient_clipping=1.0,
                gradient_accumulate=4,
                valid_metric="accuracy",
                valid_higher_better=True,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
            evaluate=dict(),
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
        By default call :obj:`voxceleb1_for_sid` with :code:`**prepare_data`

        Args:
            prepare_data (dict): same in :obj:`default_config`, support arguments in :obj:`voxceleb1_for_sid`
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
            label                 (str) - a string label of the waveform
            start_sec             (float) - optional, load the waveform from :code:`start_sec` seconds. If not presented or is :code:`math.nan`, load from the beginning.
            end_sec               (float) - optional, load the waveform from :code:`end_sec` seconds. If not presented or is :code:`math.nan`, load to the end.
            ====================  ====================
        """
        return voxceleb1_for_sid(
            **self._get_current_arguments(flatten_dict="prepare_data")
        )

    def build_encoder(
        self,
        build_encoder: dict,
        target_dir: str,
        cache_dir: str,
        train_csv_path: str,
        valid_csv_path: str,
        test_csv_paths: list,
        get_path_only: bool = False,
    ):
        """
        Build the encoder (for the labels) given the data metadata, and return the saved encoder path.
        By default generate and save a :obj:`s3prl.dataio.encoder.CategoryEncoder` from the :code:`label` column of all the csv files.

        Args:
            build_encoder (dict): same in :obj:`default_config`, no argument supported for now
            target_dir (str): Save your encoder into this directory
            cache_dir (str): If the preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            train_csv_path (str): the train path from :obj:`prepare_data`
            valid_csv_path (str): the valid path from :obj:`prepare_data`
            test_csv_paths (List[str]): the test paths from :obj:`prepare_data`
            get_path_only (str): Directly return the filepaths no matter they exist or not.

        Returns:
            str

            encoder_path: The encoder should be saved in the pickle format
        """
        encoder_path = Path(target_dir) / "encoder.pkl"
        if get_path_only:
            return encoder_path

        train_csv = pd.read_csv(train_csv_path)
        valid_csv = pd.read_csv(valid_csv_path)
        test_csvs = [pd.read_csv(path) for path in test_csv_paths]
        all_csv = pd.concat([train_csv, valid_csv, *test_csvs])

        labels = all_csv["label"].tolist()
        encoder = CategoryEncoder(labels)
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)

        return encoder_path

    def build_dataset(
        self,
        build_dataset: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        encoder_path: str,
        frame_shift: int,
    ):
        """
        Build the dataset for train/valid/test.

        Args:
            build_dataset (dict): same in :obj:`default_config`. with :code:`train`, :code:`valid`, :code:`test` keys, each
                is a dictionary with the following supported options:

                ====================  ====================
                key                   description
                ====================  ====================
                max_secs              (float) - If a waveform is longer than :code:`max_secs` seconds, randomly crop the waveform into :code:`max_secs` seconds
                sox_effects           (List[List[str]]) - If not None, apply sox effects on the utterance
                ====================  ====================

            target_dir (str): Current experiment directory
            cache_dir (str): If the preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            mode (str): train/valid/test
            data_csv (str): The metadata csv file for the specific :code:`mode`
            encoder_path (str): The pickled encoder path for encoding the labels

        Returns:
            torch Dataset

            For all train/valid/test mode, the dataset should return each item as a dictionary
            containing the following keys:

            ====================  ====================
            key                   description
            ====================  ====================
            x                     (torch.FloatTensor) - the waveform in (seq_len, 1)
            x_len                 (int) - the waveform length :code:`seq_len`
            class_id              (int) - the encoded class id
            label                 (str) - the class name
            unique_name           (str) - the unique id for this datapoint
            ====================  ====================
        """

        @dataclass
        class Config:
            train: dict = None
            valid: dict = None
            test: dict = None

        conf = Config(**build_dataset)

        assert mode in ["train", "valid", "test"]
        if mode == "train":
            conf = conf.train or {}
        elif mode == "valid":
            conf = conf.valid or {}
        elif mode == "test":
            conf = conf.test or {}

        @dataclass
        class SplitConfig:
            max_secs: float = None
            sox_effects: List[List[str]] = None

        conf = SplitConfig(**conf)

        csv = pd.read_csv(data_csv)

        start_secs = None
        if "start_sec" in csv.columns:
            start_secs = csv["start_sec"].tolist()
            start_secs = [None if math.isnan(sec) else sec for sec in start_secs]

        end_secs = None
        if "end_sec" in csv.columns:
            end_secs = csv["end_sec"].tolist()
            end_secs = [None if math.isnan(sec) else sec for sec in end_secs]

        audio_loader = LoadAudio(
            csv["wav_path"].tolist(),
            start_secs,
            end_secs,
            max_secs=conf.max_secs,
            sox_effects=conf.sox_effects,
        )

        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)

        label_encoder = EncodeCategory(csv["label"].tolist(), encoder)
        ids = csv["id"].tolist()

        class Dataset:
            def __len__(self):
                return len(ids)

            def __getitem__(self, index: int):
                audio = audio_loader[index]
                label = label_encoder[index]
                return {
                    "x": audio["wav"],
                    "x_len": audio["wav_len"],
                    "label": label["label"],
                    "class_id": label["class_id"],
                    "unique_name": ids[index],
                }

        dataset = Dataset()
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

        def _superb_sid_batch_sampler(
            train: dict = None, valid: dict = None, test: dict = None
        ):
            train = train or {}
            valid = valid or {}
            test = test or {}

            if mode == "train":
                sampler = FixedBatchSizeBatchSampler(dataset, **train)
            elif mode == "valid":
                sampler = FixedBatchSizeBatchSampler(dataset, **valid)
            elif mode == "test":
                sampler = FixedBatchSizeBatchSampler(dataset, **test)

            return sampler

        return _superb_sid_batch_sampler(**build_batch_sampler)

    def build_downstream(
        self,
        build_downstream: dict,
        downstream_input_size: int,
        downstream_output_size: int,
        downstream_input_stride: int,
    ):
        """
        Return the task-specific downstream model.
        By default build the :obj:`MeanPoolingLinear` model

        Args:
            build_downstream (dict): same in :obj:`default_config`, support arguments of :obj:`MeanPoolingLinear`
            downstream_input_size (int): the required input size of the model
            downstream_output_size (int): the required output size of the model
            downstream_input_stride (int): the input feature's stride (from 16 KHz)

        Returns:
            :obj:`s3prl.nn.interface.AbsUtteranceModel`
        """
        model = MeanPoolingLinear(
            downstream_input_size, downstream_output_size, **build_downstream
        )
        return model
