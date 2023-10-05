"""
The setting of Superb KS

Authors
  * Yist Y. Lin 2021
  * Leo 2022
"""

import logging
import pickle
from pathlib import Path
from typing import OrderedDict

import pandas as pd
from omegaconf import MISSING
from torch.utils.data import Dataset

from s3prl.dataio.corpus.speech_commands import SpeechCommandsV1
from s3prl.dataio.encoder.category import CategoryEncoder
from s3prl.dataio.sampler import BalancedWeightedSampler, FixedBatchSizeBatchSampler
from s3prl.nn.linear import MeanPoolingLinear

from .superb_sid import SuperbSID

logger = logging.getLogger(__name__)


__all__ = [
    "gsc1_for_classification",
    "SuperbKS",
]


def gsc1_for_classification(
    target_dir: str,
    cache_dir: str,
    gsc1: str,
    gsc1_test: str,
    get_path_only: bool = False,
):
    """
    Prepare Google Speech Command for classfication task
    following :obj:`SuperbKS.prepare_data` format.

    Args:
        gsc1 (str): The root path of the Google Speech Command V1 training set
        gsc1_test (str): The root path of the Google Speech Command V1 test set
        **others: refer to :obj:`SuperbKS.prepare_data`
    """
    target_dir = Path(target_dir)

    train_path = target_dir / f"train.csv"
    valid_path = target_dir / f"valid.csv"
    test_paths = [target_dir / f"test.csv"]

    if get_path_only:
        return train_path, valid_path, test_paths

    def gsc_v1_for_superb(gsc1: str, gsc1_test: str):
        corpus = SpeechCommandsV1(gsc1, gsc1_test)

        def format_fields(data: dict):
            import torchaudio

            formated_data = OrderedDict()
            for key, value in data.items():
                data_point = {
                    "wav_path": value["wav_path"],
                    "label": value["class_name"],
                    "start_sec": None,
                    "end_sec": None,
                }
                if value["class_name"] == "_silence_":
                    # NOTE: for silence, crop into 1-second segments, which
                    # is the standard way reported in the original paper
                    info = torchaudio.info(value["wav_path"])
                    for start in list(range(info.num_frames))[:: info.sample_rate]:
                        seg = data_point.copy()
                        end = min(start + 1 * info.sample_rate, info.num_frames)
                        seg["start_sec"] = start / info.sample_rate
                        seg["end_sec"] = end / info.sample_rate
                        formated_data[f"{key}_{start}_{end}"] = seg
                else:
                    formated_data[key] = data_point
            return formated_data

        train_data, valid_data, test_data = corpus.data_split
        return (
            format_fields(train_data),
            format_fields(valid_data),
            format_fields(test_data),
        )

    train_data, valid_data, test_data = gsc_v1_for_superb(gsc1, gsc1_test)

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


class SuperbKS(SuperbSID):
    def default_config(self) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=None,
            remove_all_cache=False,
            prepare_data=dict(
                gsc1=MISSING,
                gsc1_test=MISSING,
            ),
            build_encoder=dict(),
            build_dataset=dict(
                train=dict(
                    sox_effects=[
                        ["channels", "1"],
                        ["rate", "16000"],
                        ["gain", "-3.0"],
                    ],
                ),
                valid=dict(
                    sox_effects=[
                        ["channels", "1"],
                        ["rate", "16000"],
                        ["gain", "-3.0"],
                    ],
                ),
                test=dict(
                    sox_effects=[
                        ["channels", "1"],
                        ["rate", "16000"],
                        ["gain", "-3.0"],
                    ],
                ),
            ),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=32,
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
                save_step=1000,
                gradient_clipping=1.0,
                gradient_accumulate=1,
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
        By default call :obj:`gsc1_for_classification` with :code:`**prepare_data`

        Args:
            prepare_data (dict): same in :obj:`default_config`,
                support arguments in :obj:`gsc1_for_classification`
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
        return gsc1_for_classification(
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

            tokenizer_path: The tokenizer should be saved in the pickle format
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

        return encoder

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
        By default for train and valid, use :obj:`BalancedWeightedSampler`; for test use
        :obj:`FixedBatchSizeBatchSampler`

        Args:
            build_batch_sampler (dict): same in :obj:`default_config`

                ====================  ====================
                key                   description
                ====================  ====================
                train                 (dict) - arguments for :obj:`BalancedWeightedSampler`
                valid                 (dict) - arguments for :obj:`BalancedWeightedSampler`
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
            train = train or {}
            valid = valid or {}
            test = test or {}

            csv = pd.read_csv(data_csv)
            labels = csv["label"].tolist()

            if mode == "train":
                return BalancedWeightedSampler(labels, **train)
            elif mode == "valid":
                return BalancedWeightedSampler(labels, **valid)
            elif mode == "test":
                return FixedBatchSizeBatchSampler(csv, **test)

        return _build_batch_sampler(**build_batch_sampler)

    def build_downstream(
        self,
        build_downstream: dict,
        downstream_input_size: int,
        downstream_output_size: int,
        downstream_downsample_rate: int,
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
