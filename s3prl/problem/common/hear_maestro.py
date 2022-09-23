import csv
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torchaudio
from omegaconf import MISSING

from .hear_dcase_2016_task2 import HearDcase2016Task2

MAESTRO_NUM_FOLDS = 5


def prepare_maestro(
    target_dir: str,
    cache_dir: str,
    dataset_root: str,
    test_fold: int = 0,
    get_path_only: bool = False,
):
    target_dir: Path = Path(target_dir)
    train_csv = target_dir / "train.csv"
    valid_csv = target_dir / "valid.csv"
    test_csv = target_dir / "test.csv"
    if get_path_only:
        return train_csv, valid_csv, [test_csv]

    assert test_fold < MAESTRO_NUM_FOLDS, (
        f"MAESTRO only has {MAESTRO_NUM_FOLDS} folds but get 'test_fold' "
        f"arguments {test_fold}"
    )
    dataset_root = Path(dataset_root)
    wav_root = dataset_root / "16000"

    NUM_FOLD = 5
    test_id = test_fold
    valid_id = (test_fold + 1) % NUM_FOLD
    train_ids = [idx for idx in range(NUM_FOLD) if idx not in [test_id, valid_id]]

    fold_metas = []
    fold_dfs = []
    for fold_id in range(NUM_FOLD):
        with open(dataset_root / f"fold{fold_id:2d}.json".replace(" ", "0")) as f:
            metadata = json.load(f)
            fold_metas.append(metadata)

        data = defaultdict(list)
        for utt in metadata:
            wav_path = (
                wav_root / f"fold{fold_id:2d}".replace(" ", "0") / utt
            ).resolve()
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

        fold_dfs.append(pd.DataFrame(data=data))

    test_meta, test_data = fold_metas[test_id], fold_dfs[test_id]
    valid_meta, valid_data = fold_metas[valid_id], fold_dfs[valid_id]
    train_meta, train_data = {}, []
    for idx in train_ids:
        train_meta.update(fold_metas[idx])
        train_data.append(fold_dfs[idx])
    train_data: pd.DataFrame = pd.concat(train_data)

    train_data.to_csv(train_csv, index=False)
    valid_data.to_csv(valid_csv, index=False)
    test_data.to_csv(test_csv, index=False)

    return train_csv, valid_csv, [test_csv]


class HearMaestro(HearDcase2016Task2):
    def default_config(self) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=None,
            remove_all_cache=False,
            prepare_data=dict(
                dataset_root=MISSING,
                test_fold=MISSING,
            ),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=5,
                    shuffle=True,
                ),
                valid=dict(
                    item="record_id",
                ),
                test=dict(
                    item="record_id",
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
            ),
            build_model=dict(
                upstream_trainable=False,
            ),
            build_task=dict(
                prediction_type="multilabel",
                scores=["event_onset_50ms_fms", "event_onset_offset_50ms_20perc_fms"],
                postprocessing_grid={
                    "median_filter_ms": [150],
                    "min_duration": [50],
                },
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
                total_steps=15000,
                log_step=100,
                eval_step=500,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate=1,
                valid_metric="event_onset_50ms_fms",
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
        return prepare_maestro(
            **self._get_current_arguments(flatten_dict="prepare_data")
        )
