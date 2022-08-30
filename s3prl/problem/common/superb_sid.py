import logging
import math
import pickle
from pathlib import Path
from typing import List, OrderedDict

import pandas as pd
from omegaconf import MISSING
from torch.utils.data import Dataset

from s3prl.corpus.voxceleb1sid import voxceleb1_for_utt_classification
from s3prl.nn.interface import AbsFeaturizer, AbsUpstream, AbsUtteranceModel
from s3prl.nn.upstream import Featurizer, S3PRLUpstream, UpstreamDownstreamModel
from s3prl.sampler import FixedBatchSizeBatchSampler

from .run import Common

logger = logging.getLogger(__name__)


class SuperbSID(Common):
    @classmethod
    def default_config(cls) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=str(Path.home() / ".cache" / "s3prl" / "data"),
            remove_all_cache=False,
            prepare_data=dict(dataset_root=MISSING),
            build_encoder=dict(),
            build_dataset=dict(
                max_secs=8.0,
            ),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=8,
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
                gradient_accumulate_steps=4,
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
        _get_path_only=False,
    ):
        target_dir = Path(_target_dir)

        train_path = target_dir / f"train.csv"
        valid_path = target_dir / f"valid.csv"
        test_paths = [target_dir / f"test.csv"]

        if _get_path_only:
            return train_path, valid_path, test_paths

        train_data, valid_data, test_data = voxceleb1_for_utt_classification(
            dataset_root
        ).values()

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
        max_secs: float,
    ):
        """
        _mode is in ["train", "valid", "test"]
        """
        from s3prl.dataset.common_pipes import RandomCrop

        EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]

        from s3prl.dataset.utterance_classification_pipe import (
            UtteranceClassificationPipe,
        )

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
        )(data_points, category=encoder)
        dataset = RandomCrop(max_secs=max_secs)(dataset)
        dataset.update_output_keys(dict(x="wav_crop", x_len="wav_crop_len"))

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
        Feed the single hidden state to the downstream model
        """
        from s3prl.nn.linear import MeanPoolingLinear

        model = MeanPoolingLinear(
            _downstream_input_size, _downstream_output_size, hidden_size
        )
        return model

    @classmethod
    def build_model(
        cls,
        _model_output_size: str,
        _build_upstream: dict,
        _build_featurizer: dict,
        _build_downstream: dict,
        upstream_trainable: bool,
    ) -> AbsUtteranceModel:
        upstream = cls.build_upstream(**_build_upstream)
        featurizer: Featurizer = cls.build_featurizer(upstream, **_build_featurizer)
        downstream = cls.build_downstream(
            featurizer.output_size,
            _model_output_size,
            featurizer.downsample_rate,
            **_build_downstream,
        )
        model = UpstreamDownstreamModel(
            upstream, featurizer, downstream, upstream_trainable
        )
        return model
