import json
import pickle
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from omegaconf import MISSING

from s3prl.dataio.encoder import CategoryEncoder
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler
from s3prl.dataset.utterance_classification_pipe import HearScenePipe
from s3prl.nn.hear import HearFullyConnectedPrediction
from s3prl.task.scene_prediction import ScenePredictionTask

from .superb_sid import SuperbSID


def hear_scene_trainvaltest(
    target_dir: str,
    cache_dir: str,
    dataset_root: str,
    get_path_only: bool = False,
):
    target_dir = Path(target_dir)
    dataset_root = Path(dataset_root)
    wav_root: Path = dataset_root / "16000"

    train_csv = target_dir / "train.csv"
    valid_csv = target_dir / "valid.csv"
    test_csv = target_dir / "test_csv"

    if get_path_only:
        return train_csv, valid_csv, [test_csv]

    def load_json(filepath):
        with open(filepath, "r") as fp:
            return json.load(fp)

    def split_to_df(split: str) -> pd.DataFrame:
        meta = load_json(dataset_root / f"{split}.json")
        data = defaultdict(list)
        for k in list(meta.keys()):
            data["id"].append(k)
            data["wav_path"].append(wav_root / split / k)
            data["labels"].append(",".join([str(label).strip() for label in meta[k]]))
        return pd.DataFrame(data=data)

    split_to_df("train").to_csv(train_csv, index=False)
    split_to_df("valid").to_csv(valid_csv, index=False)
    split_to_df("test").to_csv(test_csv, index=False)

    return train_csv, valid_csv, [test_csv]


class HearFSD(SuperbSID):
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
            build_batch_sampler=dict(
                train=dict(
                    batch_size=10,
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
                hidden_layers=2,
                pooling_type="MeanPooling",
            ),
            build_model=dict(
                upstream_trainable=False,
            ),
            build_task=dict(
                prediction_type="multilabel",
                scores=["mAP", "top1_acc", "d_prime", "aucroc"],
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
                total_steps=40000,
                log_step=100,
                eval_step=1000,
                save_step=100,
                gradient_clipping=1.0,
                gradient_accumulate=4,
                valid_metric="mAP",
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
        return hear_scene_trainvaltest(
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
        encoder_path = Path(target_dir) / "encoder.pkl"
        if get_path_only:
            return encoder_path

        train_csv = pd.read_csv(train_csv_path)
        valid_csv = pd.read_csv(valid_csv_path)
        test_csvs = [pd.read_csv(path) for path in test_csv_paths]
        all_csv = pd.concat([train_csv, valid_csv, *test_csvs])
        all_labels = []
        for rowid, row in all_csv.iterrows():
            labels = str(row["labels"]).split(",")
            labels = [l.strip() for l in labels]
            all_labels.extend(labels)

        encoder = CategoryEncoder(all_labels)
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
        df = pd.read_csv(data_csv)
        data = OrderedDict()
        for rowid, row in df.iterrows():
            data[row["id"]] = dict(
                wav_path=row["wav_path"],
                labels=[label.strip() for label in str(row["labels"]).split(",")],
            )

        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)

        dataset = HearScenePipe()(data, tools={"category": encoder})
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
        @dataclass
        class Config:
            train: dict = None
            valid: dict = None
            test: dict = None

        conf = Config(**build_batch_sampler)
        return FixedBatchSizeBatchSampler(dataset, **(conf.train or {}))

    def build_downstream(
        self,
        build_downstream: dict,
        downstream_input_size: int,
        downstream_output_size: int,
        downstream_input_stride: int,
    ):
        return HearFullyConnectedPrediction(
            downstream_input_size, downstream_output_size, **build_downstream
        )

    def build_task(
        self,
        build_task: dict,
        model: torch.nn.Module,
        encoder,
        valid_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
    ):
        return ScenePredictionTask(model, encoder, **build_task)
