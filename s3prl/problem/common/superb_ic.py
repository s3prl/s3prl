"""
The setting of Superb IC

Authors
  * Wei-Cheng Tseng 2021
  * Leo 2021
  * Leo 2022
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
import torch
from omegaconf import MISSING
from torch.utils.data import Dataset

from s3prl.dataio.corpus.fluent_speech_commands import FluentSpeechCommands
from s3prl.dataio.dataset import EncodeCategories, LoadAudio
from s3prl.dataio.encoder.category import CategoryEncoders
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler
from s3prl.nn.linear import MeanPoolingLinear
from s3prl.task.utterance_classification_task import (
    UtteranceMultiClassClassificationTask,
)

from .run import Common

logger = logging.getLogger(__name__)


__all__ = [
    "fsc_for_multi_classification",
    "SuperbIC",
]


def fsc_for_multi_classification(
    target_dir: str,
    cache_dir: str,
    dataset_root: str,
    n_jobs: int = 6,
    get_path_only: bool = False,
):
    """
    Prepare Fluent Speech Command for multi-class classfication
    following :obj:`SuperbIC.prepare_data` format. The standard usage
    is to use three labels jointly: action, object, and location.

    Args:
        dataset_root (str): The root path of Fluent Speech Command
        n_jobs (int): to speed up the corpus parsing procedure
    """
    target_dir = Path(target_dir)

    train_path = target_dir / f"train.csv"
    valid_path = target_dir / f"valid.csv"
    test_paths = [target_dir / f"test.csv"]

    if get_path_only:
        return train_path, valid_path, test_paths

    def format_fields(data_points: dict):
        return {
            key: dict(
                wav_path=value["path"],
                labels=f"{value['action']} ; {value['object']} ; {value['location']}",
            )
            for key, value in data_points.items()
        }

    corpus = FluentSpeechCommands(dataset_root, n_jobs)
    train_data, valid_data, test_data = corpus.data_split
    train_data = format_fields(train_data)
    valid_data = format_fields(valid_data)
    test_data = format_fields(test_data)

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


class SuperbIC(Common):
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
            build_dataset=dict(),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=32,
                    shuffle=True,
                ),
                valid=dict(
                    batch_size=32,
                ),
                test=dict(
                    batch_size=32,
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
                log_step=100,
                eval_step=5000,
                save_step=250,
                gradient_clipping=1.0,
                gradient_accumulate=1,
                valid_metric="accuracy",
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
        By default call :obj:`fsc_for_multi_classification` with :code:`**prepare_data`

        Args:
            prepare_data (dict): same in :obj:`default_config`,
                arguments for :obj:`fsc_for_multi_classification`
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
            labels                (str) - the string labels of the waveform, separated by a ';'
            ====================  ====================

            The number of the label columns can be arbitrary.
        """
        return fsc_for_multi_classification(
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
        By default generate and save a :obj:`s3prl.dataio.encoder.CategoryEncoders` from all the columns
        prefixing :code:`label` from all the csv files.

        Args:
            build_encoder (dict): same in :obj:`default_config`, no argument supported for now
            target_dir (str): Save your encoder into this directory
            cache_dir (str): If the preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            train_csv_path (str): the train path from :obj:`prepare_data`
            valid_csv_path (str): the valid path from :obj:`prepare_data`
            test_csv_paths (List[str]): the test paths from :obj:`prepare_data`
            get_path_only (bool): Directly return the filepaths no matter they exist or not.

        Returns:
            str

            tokenizer_path: The tokenizer should be saved in the pickle format
        """
        encoder_path = Path(target_dir) / "encoder.pkl"
        if get_path_only:
            return encoder_path

        train_csv = pd.read_csv(train_csv_path)
        valid_csv = pd.read_csv(valid_csv_path)
        test_csvs = [pd.read_csv(path) for path in test_csv_paths]
        all_csv = pd.concat([train_csv, valid_csv, *test_csvs])

        multilabels = [
            [label.strip() for label in multilabel.split(";")]
            for multilabel in all_csv["labels"].tolist()
        ]
        encoder = CategoryEncoders(
            [single_category_labels for single_category_labels in zip(*multilabels)]
        )
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)

        return encoder

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
            build_dataset (dict): same in :obj:`default_config`, no argument supported for now
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
            class_ids             (torch.LongTensor) - the encoded class ids. shape: (num_class, )
            labels                (List[str]) - the class name. length: num_class
            unique_name           (str) - the unique id for this datapoint
            ====================  ====================
        """
        csv = pd.read_csv(data_csv)
        ids = csv["id"].tolist()

        audio_loader = LoadAudio(csv["wav_path"].tolist())

        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)

        label_encoder = EncodeCategories(
            [
                [label.strip() for label in multilabel.split(";")]
                for multilabel in csv["labels"].tolist()
            ],
            encoder,
        )

        class Dataset:
            def __len__(self):
                return len(audio_loader)

            def __getitem__(self, index: int):
                audio = audio_loader[index]
                label = label_encoder[index]
                return {
                    "x": audio["wav"],
                    "x_len": audio["wav_len"],
                    "class_ids": label["class_ids"],
                    "labels": label["labels"],
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
        dataset: Dataset,
    ):
        """
        Return the batch sampler for torch DataLoader.
        By default call :obj:`superb_sid_batch_sampler` with :code:`**build_batch_sampler`.

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

        def _build_batch_sampler(
            train: dict = None, valid: dict = None, test: dict = None
        ):
            if mode == "train":
                return FixedBatchSizeBatchSampler(dataset, **train)
            elif mode == "valid":
                return FixedBatchSizeBatchSampler(dataset, **valid)
            elif mode == "test":
                return FixedBatchSizeBatchSampler(dataset, **test)

        return _build_batch_sampler(**build_batch_sampler)

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
            build_downstream (dict): same in :obj:`default_config`,
                support arguments of :obj:`MeanPoolingLinear`
            downstream_input_size (int): the required input size of the model
            downstream_output_size (int): the required output size of the model
            downstream_input_stride (int): the input feature's stride (from 16 KHz)

        Returns:
            :obj:`AbsUtteranceModel`
        """
        model = MeanPoolingLinear(
            downstream_input_size, downstream_output_size, **build_downstream
        )
        return model

    def build_task(
        self,
        build_task: dict,
        model: torch.nn.Module,
        encoder,
        valid_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
    ):
        """
        Build the task, which defines the logics for every train/valid/test forward step for the :code:`model`,
        and the logics for how to reduce all the batch results from multiple train/valid/test steps into metrics

        By default build :obj:`UtteranceMultiClassClassificationTask`

        Args:
            build_task (dict): same in :obj:`default_config`, no argument supported for now
            model (torch.nn.Module): the model built by :obj:`build_model`
            encoder: the encoder built by :obj:`build_encoder`
            valid_df (pd.DataFrame): metadata of the valid set
            test_df (pd.DataFrame): metadata of the test set

        Returns:
            Task
        """
        return UtteranceMultiClassClassificationTask(model, encoder)
