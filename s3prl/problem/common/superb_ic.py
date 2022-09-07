import logging
import pickle
from ast import literal_eval
from pathlib import Path
from typing import OrderedDict

import pandas as pd
from omegaconf import MISSING
from torch.utils.data import Dataset

from s3prl.corpus.fluent_speech_commands import FluentSpeechCommands
from s3prl.dataset.utterance_classification_pipe import (
    UtteranceMultipleCategoryClassificationPipe,
)
from s3prl.encoder.category import CategoryEncoders
from s3prl.nn.interface import AbsUtteranceModel
from s3prl.nn.linear import MeanPoolingLinear
from s3prl.sampler import FixedBatchSizeBatchSampler
from s3prl.task.utterance_classification_task import (
    UtteranceMultiClassClassificationTask,
)

from .run import Common

logger = logging.getLogger(__name__)


class SuperbIC(Common):
    @classmethod
    def default_config(cls) -> dict:
        """
        Args:
            start (int):
                The starting stage of the problem script.
                Default: 0
            stop (int):
                The stoping stage of the problem script, set `None` to reach the final stage.
                Default: None
            target_dir (str):
                The directory that stores the script result.
                Default: MISSING
            cache_dir (str):
                The directory that caches the processed data.
                Default: /home/user/.cache/s3prl/data
            remove_all_cache (bool):
                Whether to remove all the cache stored under `cache_dir`.
                Default: False
            prepare_data (dict):
                The dict that stores the prepare data config.
                Default: dict(dataset_root=MISSING)
            build_encoder (dict):
                The dict that stores the encoder config.
                Default: dict()
            build_dataset (dict):
                The dict that stores the dataset config.
                Default: dict()
            build_batch_sampler (dict):
                The dict that stores the batch sampler config.
                Default: dict(
                    train=dict(batch_size=32,shuffle=True,),
                    valid=dict(batch_size=32,),
                    test=dict(batch_size=32,),
                )
            build_upstream (dict):
                The dict that stores the upstream config.
                Default: dict(name="fbank",)
            build_featurizer (dict):
                The dict that stores the featurizer config.
                Default: dict(layer_selections=None, normalize=False,)
            build_downstream (dict):
                The dict that stores the downstream config.
                Default: dict(hidden_size=256,)
            build_model (dict):
                The dict that stores the model config.
                Default: dict(upstream_trainable=False,)
            build_task (dict):
                The dict that stores the task config.
                Default: dict()
            build_optimizer (dict):
                The dict that stores the optimizer config.
                Default: dict(
                    name="Adam",
                    conf=dict(lr=1.0e-4,),
                )
            build_scheduler (dict):
                The dict that stores the scheduler config.
                Default: dict(name="ExponentialLR",gamma=0.9,)
            save_model (dict):
                The dict that stores the save model config.
                Default: dict()
            save_task (dict):
                The dict that stores the save task config.
                Default: dict()
            train (dict):
                The dict that stores the training script config.
                Default: dict(
                    total_steps=200000,
                    log_step=100,
                    eval_step=2000,
                    save_step=500,
                    gradient_clipping=1.0,
                    gradient_accumulate_steps=1,
                    valid_metric="accuracy",
                    valid_higher_better=True,
                    auto_resume=True,
                    resume_ckpt_dir=None,
                )
            evaluate (dict):
                The dict that stores the evaluation script config.
                Default: dict()

        Return:
            default_config (dict): The dict that stores all the config
        """
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=str(Path.home() / ".cache" / "s3prl" / "data"),
            remove_all_cache=False,
            prepare_data=dict(dataset_root=MISSING),
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
                name="fbank",
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
                eval_step=2000,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate_steps=1,
                valid_metric="accuracy",
                valid_higher_better=True,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
            evaluate=dict(),
        )

    @classmethod
    def prepare_data(
        cls,
        _target_dir,
        _cache_dir,
        dataset_root: str,
        n_jobs: int = 6,
        _get_path_only=False,
    ):
        """
        Args:
            _target_dir (str):
                The directory that stores the script result.
            _cache_dir (str):
                The directory that caches the processed data.
            dataset_root (str):
                The root directory of the input dataset.
            n_jobs (int):
                The number of jobs to use for the computation.
                Default: 6
            _get_path_only (bool):
                Whether to prepare only the path and not the actual data.
                Default: False

        Return:
            train_path (str):
                The path that points to the prepared train csv file.
                Default: "`_target_dir`/train.csv"
            valid_path (str):
                The path that points to the prepared valid csv file.
                Default: "`_target_dir`/valid.csv"
            test_paths (list[str]):
                The path that points to the prepared test csv files.
                Default: ["`_target_dir`/test.csv",]
        """
        target_dir = Path(_target_dir)

        train_path = target_dir / f"train.csv"
        valid_path = target_dir / f"valid.csv"
        test_paths = [target_dir / f"test.csv"]

        if _get_path_only:
            return train_path, valid_path, test_paths

        def format_fields(data_points: dict):
            return {
                key: dict(
                    wav_path=value["path"],
                    labels=[value["action"], value["object"], value["location"]],
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

    @classmethod
    def build_encoder(
        cls,
        _target_dir,
        _cache_dir,
        _train_csv_path,
        _valid_csv_path,
        _test_csv_paths,
        _get_path_only=False,
    ):
        """
        Args:
            _target_dir (str):
                The directory that stores the script result.
            _cache_dir (str):
                The directory that caches the processed data.
            _train_csv_path (str):
                The path that points to the prepared train csv file.
            _valid_csv_path (str):
                The path that points to the prepared valid csv file.
            _test_csv_paths (str):
                The path that points to the prepared test csv files.
            _get_path_only (bool):
                Whether to prepare only the path and not the actual data.
                Default: False

        Return:
            encoder (CategoryEncoders):
                The builded encoder object.
        """
        encoder_path = Path(_target_dir) / "encoder.pkl"
        if _get_path_only:
            return encoder_path

        train_csv = pd.read_csv(_train_csv_path)
        valid_csv = pd.read_csv(_valid_csv_path)
        test_csvs = [pd.read_csv(path) for path in _test_csv_paths]
        all_csv = pd.concat([train_csv, valid_csv, *test_csvs])

        labels = all_csv["labels"].apply(literal_eval).tolist()
        encoder = CategoryEncoders(
            [list(sorted(set((label)))) for label in zip(*labels)]
        )
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)

        return encoder

    @classmethod
    def build_dataset(
        cls,
        _target_dir: str,
        _cache_dir: str,
        _mode: str,
        _data_csv: str,
        _encoder_path: str,
    ):
        """
        Args:
            _target_dir (str):
                The directory that stores the script result.
            _cache_dir (str):
                The directory that caches the processed data.
            _mode (str):
                The mode of the dataset.
                Default choices: ("train", "valid", "test")
            _data_csv (str):
                The path that points to the prepared (train/valid/test) csv file.
            _encoder_path (str):
                The path that points to the stored encoder object file.

        Return:
            dataset (UtteranceMultipleCategoryClassificationPipe):
                The builded dataset object.
        """
        data_points = OrderedDict()
        csv = pd.read_csv(_data_csv)
        csv["labels"] = csv["labels"].apply(literal_eval)
        for _, row in csv.iterrows():
            data_points[row["id"]] = {
                "wav_path": row["wav_path"],
                "labels": row["labels"],
            }

        with open(_encoder_path, "rb") as f:
            encoder = pickle.load(f)

        dataset = UtteranceMultipleCategoryClassificationPipe(
            train_category_encoder=False
        )(
            data_points,
            tools={"categories": encoder},
        )
        return dataset

    @classmethod
    def build_batch_sampler(
        cls,
        _target_dir: str,
        _cache_dir: str,
        _mode: str,
        _data_csv: str,
        _dataset: Dataset,
        train: dict = {},
        valid: dict = {},
        test: dict = {},
    ):
        """
        Args:
            _target_dir (str):
                The directory that stores the script result.
            _cache_dir (str):
                The directory that caches the processed data.
            _mode (str):
                The mode of the dataset.
                Default choices: ("train", "valid", "test")
            _data_csv (str):
                The path that points to the prepared (train/valid/test) csv file.
            _dataset (Dataset):
                The dataset object.
            train (dict):
                The args for the batch sampler during "train" `_mode`.
                Default: dict()
            valid (dict):
                The args for the batch sampler during "valid" `_mode`.
                Default: dict()
            test (dict):
                The args for the batch sampler during "test" `_mode`.
                Default: dict()

        Return:
            sampler (FixedBatchSizeBatchSampler):
                The builded batch sampler object depended on the given `_mode`.
        """
        if _mode == "train":
            sampler = FixedBatchSizeBatchSampler(_dataset, **train)
        elif _mode == "valid":
            sampler = FixedBatchSizeBatchSampler(_dataset, **valid)
        elif _mode == "test":
            sampler = FixedBatchSizeBatchSampler(_dataset, **test)

        return sampler

    @classmethod
    def build_downstream(
        cls,
        _downstream_input_size: int,
        _downstream_output_size: int,
        _downstream_downsample_rate: int,
        hidden_size: int,
    ) -> AbsUtteranceModel:
        """
        Args:
            _downstream_input_size (int):
                The input size of the downstream model.
            _downstream_output_size (int):
                The output size of the downstream model.
            _downstream_downsample_rate (int):
                The downstream downsample rate.
            hidden_size (int):
                The hidden state size of the downstream model.

        Return:
            model (AbsUtteranceModel):
                The builded downstream model object.
                Defualt: MeanPoolingLinear
        """
        model = MeanPoolingLinear(
            _downstream_input_size, _downstream_output_size, hidden_size
        )
        return model

    @classmethod
    def build_task(cls, _model, _encoder):
        """
        Args:
            _model (UtteranceClassifierExample):
                The utterance classifier model for the task.
            _encoder (CategoryEncoders):
                The category encoder model for the task.

        Return:
            task (UtteranceMultiClassClassificationTask):
                The builded task object.
        """
        return UtteranceMultiClassClassificationTask(_model, _encoder)
