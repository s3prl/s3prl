import json
import logging
import pickle
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from omegaconf import MISSING

from s3prl.dataio.dataset import FrameLabelDataset, get_info
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler, GroupSameItemSampler
from s3prl.task.event_prediction import EventPredictionTask

from ._hear_util import resample_hear_corpus
from .hear_fsd import HearFSD

logger = logging.getLogger(__name__)

__all__ = [
    "HearDcase2016Task2",
]


def dcase_2016_task2(
    target_dir: str,
    cache_dir: str,
    dataset_root: str,
    get_path_only: bool = False,
):
    target_dir: Path = Path(target_dir)
    train_csv = target_dir / "train.csv"
    valid_csv = target_dir / "valid.csv"
    test_csv = target_dir / "test.csv"

    if get_path_only:
        return train_csv, valid_csv, [test_csv]

    resample_hear_corpus(dataset_root, target_sr=16000)

    dataset_root = Path(dataset_root)
    wav_root: Path = dataset_root / "16000"

    def json_to_csv(json_path: str, csv_path: str, split: str):
        with open(json_path) as fp:
            metadata = json.load(fp)

        data = defaultdict(list)
        for utt in metadata:
            wav_path: Path = (wav_root / split / utt).resolve()
            assert wav_path.is_file()
            info = torchaudio.info(wav_path)
            baseinfo = {
                "record_id": utt,
                "wav_path": str(wav_path),
                "duration": info.num_frames / info.sample_rate,
            }
            for segment in metadata[utt]:
                fullinfo = deepcopy(baseinfo)
                fullinfo[
                    "utt_id"
                ] = f"{baseinfo['record_id']}-{int(segment['start'])}-{int(segment['end'])}"
                fullinfo["labels"] = segment["label"]
                fullinfo["start_sec"] = segment["start"] / 1000
                fullinfo["end_sec"] = segment["end"] / 1000

                for key, value in fullinfo.items():
                    data[key].append(value)

        pd.DataFrame(data=data).to_csv(csv_path, index=False)

    json_to_csv(dataset_root / "train.json", train_csv, "train")
    json_to_csv(dataset_root / "valid.json", valid_csv, "valid")
    json_to_csv(dataset_root / "test.json", test_csv, "test")

    return train_csv, valid_csv, [test_csv]


class HearDcase2016Task2(HearFSD):
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
            build_dataset=dict(
                train=dict(
                    chunk_secs=4.0,
                    step_secs=4.0,
                ),
                valid=dict(
                    chunk_secs=4.0,
                    step_secs=4.0,
                ),
                test=dict(
                    chunk_secs=4.0,
                    step_secs=4.0,
                ),
            ),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=5,
                    shuffle=True,
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
                hidden_layers=2,
            ),
            build_model=dict(
                upstream_trainable=False,
            ),
            build_task=dict(
                prediction_type="multilabel",
                scores=["event_onset_200ms_fms", "segment_1s_er"],
                postprocessing_grid={
                    "median_filter_ms": [250],
                    "min_duration": [125, 250],
                },
            ),
            build_optimizer=dict(
                name="Adam",
                conf=dict(
                    lr=1.0e-3,
                ),
            ),
            build_scheduler=dict(
                name="ExponentialLR",
                gamma=0.9,
            ),
            save_model=dict(),
            save_task=dict(),
            train=dict(
                total_steps=15000,
                log_step=100,
                eval_step=500,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate=1,
                valid_metric="event_onset_200ms_fms",
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
        return dcase_2016_task2(
            **self._get_current_arguments(flatten_dict="prepare_data")
        )

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
        @dataclass
        class Config:
            train: dict = None
            valid: dict = None
            test: dict = None

        conf = Config(**build_dataset)
        conf = getattr(conf, mode)
        conf = conf or {}

        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)

        df = pd.read_csv(data_csv)
        df["label"] = [encoder.encode(label) for label in df["labels"].tolist()]

        dataset = FrameLabelDataset(df, len(encoder), frame_shift, **conf)
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
        if mode == "train":
            return FixedBatchSizeBatchSampler(dataset, **(conf.train or {}))
        elif mode == "valid":
            record_ids = get_info(dataset, ["record_id"], target_dir / "valid_stats")
            return GroupSameItemSampler(record_ids)
        elif mode == "test":
            record_ids = get_info(dataset, ["record_id"], target_dir / "test_stats")
            return GroupSameItemSampler(record_ids)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def build_task(
        self,
        build_task: dict,
        model: torch.nn.Module,
        encoder,
        valid_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
    ):
        def df_to_events(df: pd.DataFrame):
            data = {}
            for rowid, row in df.iterrows():
                record_id = row["record_id"]
                if not record_id in data:
                    data[record_id] = []
                data[record_id].append(
                    {
                        "start": row["start_sec"] * 1000,
                        "end": row["end_sec"] * 1000,
                        "label": row["labels"],
                    }
                )
            return data

        valid_events = None if valid_df is None else df_to_events(valid_df)
        test_events = None if test_df is None else df_to_events(test_df)

        return EventPredictionTask(
            model,
            encoder,
            valid_target_events=valid_events,
            test_target_events=test_events,
            **build_task,
        )
