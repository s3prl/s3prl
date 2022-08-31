import logging
import math
import pickle
from pathlib import Path
from typing import OrderedDict

import pandas as pd
from omegaconf import MISSING
from torch.utils.data import Dataset

from s3prl.nn.interface import AbsFrameModel
from s3prl.nn.linear import MeanPoolingLinear
from s3prl.corpus.speech_commands import SpeechCommandsV1
from s3prl.sampler import BalancedWeightedSampler, FixedBatchSizeBatchSampler
from s3prl.dataset.utterance_classification_pipe import UtteranceClassificationPipe

from .run import Common

logger = logging.getLogger(__name__)

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SuperbKS(Common):
    @classmethod
    def default_config(cls) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=str(Path.home() / ".cache" / "s3prl" / "data"),
            remove_all_cache=False,
            prepare_data=dict(
                gsc1=MISSING,
                gsc1_test=MISSING,
            ),
            build_encoder=dict(),
            build_dataset=dict(),
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
                name="fbank",
            ),
            build_featurizer=dict(
                layer_selections=None,
                normalize=False,
            ),
            build_downstream=dict(hidden_size=256),
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
                valid_higher_better=False,
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
        gsc1: str,
        gsc1_test: str,
        _get_path_only=False,
    ):
        target_dir = Path(_target_dir)

        train_path = target_dir / f"train.csv"
        valid_path = target_dir / f"valid.csv"
        test_paths = [target_dir / f"test.csv"]

        if _get_path_only:
            return train_path, valid_path, test_paths

        def gsc_v1_for_superb(gsc1: str, gsc1_test: str, n_jobs: int = 6):
            corpus = SpeechCommandsV1(gsc1, gsc1_test, n_jobs)

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
        from s3prl.encoder.category import CategoryEncoder

        encoder_path = Path(_target_dir) / "encoder.pkl"
        if _get_path_only:
            return encoder_path

        train_csv = pd.read_csv(_train_csv_path)
        valid_csv = pd.read_csv(_valid_csv_path)
        test_csvs = [pd.read_csv(path) for path in _test_csv_paths]
        all_csv = pd.concat([train_csv, valid_csv, *test_csvs])

        labels = all_csv["label"].tolist()
        encoder = CategoryEncoder(labels)
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
        data_points = OrderedDict()
        csv = pd.read_csv(_data_csv)
        for _, row in csv.iterrows():
            if "start_sec" in row and "end_sec" in row:
                start_sec = row["start_sec"]
                end_sec = row["end_sec"]

                if math.isnan(start_sec):
                    start_sec = None

                if math.isnan(end_sec):
                    end_sec = None

            else:
                start_sec = None
                end_sec = None

            data_points[row["id"]] = {
                "wav_path": row["wav_path"],
                "label": row["label"],
                "start_sec": start_sec,
                "end_sec": end_sec,
            }

        with open(_encoder_path, "rb") as f:
            encoder = pickle.load(f)

        dataset = UtteranceClassificationPipe(
            train_category_encoder=False, sox_effects=EFFECTS
        )(
            data_points,
            tools={"category": encoder},
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
        from collections import Counter

        csv = pd.read_csv(_data_csv)

        def get_weights(csv):
            labels = csv["label"].tolist()
            class2weight = Counter()
            class2weight.update(labels)

            weights = []
            for row_id, row in csv.iterrows():
                weights.append(len(csv) / class2weight[row["label"]])

            return weights

        if _mode == "train":
            sampler = BalancedWeightedSampler(csv, get_weights=get_weights, **train)
        elif _mode == "valid":
            sampler = BalancedWeightedSampler(csv, get_weights=get_weights, **valid)
        elif _mode == "test":
            sampler = FixedBatchSizeBatchSampler(csv, **test)

        return sampler

    @classmethod
    def build_downstream(
        cls,
        _downstream_input_size: int,
        _downstream_output_size: int,
        _downstream_downsample_rate: int,
        hidden_size: int,
    ) -> AbsFrameModel:
        model = MeanPoolingLinear(
            _downstream_input_size, _downstream_output_size, hidden_size
        )
        return model
