import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from omegaconf import MISSING

from .hear_fsd import HearFSD

ESC50_NUM_FOLDS = 5


def hear_scene_kfolds(
    target_dir: str,
    cache_dir: str,
    dataset_root: str,
    test_fold: int,
    num_folds: int,
    get_path_only: bool = False,
):
    assert test_fold < num_folds, (
        "test_fold id must be smaller than num_folds. "
        f"get test_fold={test_fold} and num_folds={num_folds}"
    )

    target_dir = Path(target_dir)
    train_csv = target_dir / "train.csv"
    valid_csv = target_dir / "valid.csv"
    test_csv = target_dir / "test.csv"

    if get_path_only:
        return train_csv, valid_csv, [test_csv]

    dataset_root = Path(dataset_root)
    wav_root = dataset_root / "16000"

    def load_json(filepath):
        with open(filepath, "r") as fp:
            return json.load(fp)

    fold_metas = []
    fold_datas = []
    for fold_id in range(num_folds):
        meta = load_json(dataset_root / f"fold{fold_id:2d}.json".replace(" ", "0"))
        fold_metas.append(meta)

        data = defaultdict(list)
        for k in list(meta.keys()):
            wav_path = wav_root / f"fold{fold_id:2d}".replace(" ", "0") / k
            labels = meta[k]
            data["id"].append(k)
            data["wav_path"].append(wav_path)
            data["labels"].append(",".join([str(label).strip() for label in labels]))

        df = pd.DataFrame(data=data)
        fold_datas.append(df)

    test_id = test_fold
    valid_id = (test_fold + 1) % num_folds
    train_ids = [idx for idx in range(num_folds) if idx not in [test_id, valid_id]]

    test_data = fold_datas[test_id]
    valid_data = fold_datas[valid_id]
    train_data = []
    for idx in train_ids:
        train_data.append(fold_datas[idx])
    train_data = pd.concat(train_data)

    train_data.to_csv(train_csv, index=False)
    valid_data.to_csv(valid_csv, index=False)
    test_data.to_csv(test_csv, index=False)

    return train_csv, valid_csv, [test_csv]


class HearESC50(HearFSD):
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
                num_folds=ESC50_NUM_FOLDS,
            ),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=32,
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
                prediction_type="multiclass",
                scores=["top1_acc", "d_prime", "aucroc", "mAP"],
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
                total_steps=4000,
                log_step=100,
                eval_step=500,
                save_step=100,
                gradient_clipping=1.0,
                gradient_accumulate=4,
                valid_metric="top1_acc",
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
        return hear_scene_kfolds(
            **self._get_current_arguments(flatten_dict="prepare_data")
        )
