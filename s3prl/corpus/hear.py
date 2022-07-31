import json
import torchaudio
from pathlib import Path
from collections import OrderedDict

from s3prl import Container
from s3prl.encoder.category import CategoryEncoder


def dcase_2016_task2(dataset_root: str):
    dataset_root = Path(dataset_root)
    wav_root = dataset_root / "16000"

    def load_json(filepath):
        with open(filepath, "r") as fp:
            return json.load(fp)

    train_meta = load_json(dataset_root / "train.json")
    valid_meta = load_json(dataset_root / "valid.json")
    test_meta = load_json(dataset_root / "test.json")

    def meta_to_data(meta, split: str):
        data = {}
        for utt in meta:
            wav_path: Path = wav_root / split / utt
            assert wav_path.is_file()
            info = torchaudio.info(wav_path)
            data[utt] = dict(
                wav_path=str(wav_path.resolve()),
                start_sec=0.0,
                end_sec=info.num_frames / info.sample_rate,
                segments=OrderedDict(),
            )
            for segment in meta[utt]:
                if segment["label"] not in data[utt]["segments"]:
                    data[utt]["segments"][segment["label"]] = []

                data[utt]["segments"][segment["label"]].append(
                    (segment["start"] / 1000, segment["end"] / 1000)
                )
        return data

    train_data = meta_to_data(train_meta, "train")
    valid_data = meta_to_data(valid_meta, "valid")
    test_data = meta_to_data(test_meta, "test")

    return Container(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        valid_target_events=valid_meta,
        test_target_events=test_meta,
    )


def maestro(dataset_root: str, test_fold: int = 0):
    dataset_root = Path(dataset_root)
    wav_root = dataset_root / "16000"

    def load_json(filepath):
        with open(filepath, "r") as fp:
            return json.load(fp)

    NUM_FOLD = 5
    test_id = test_fold
    valid_id = (test_fold + 1) % NUM_FOLD
    train_ids = [idx for idx in range(NUM_FOLD) if idx not in [test_id, valid_id]]

    fold_metas = []
    fold_datas = []
    for fold_id in range(NUM_FOLD):
        meta = load_json(dataset_root / f"fold{fold_id:2d}.json".replace(" ", "0"))
        fold_metas.append(meta)

        data = {}
        for k in list(meta.keys()):
            wav_path = wav_root / f"fold{fold_id:2d}".replace(" ", "0") / k
            info = torchaudio.info(wav_path)
            item = dict(
                wav_path=wav_path,
                start_sec=0.0,
                end_sec=info.num_frames / info.sample_rate,
                segments=OrderedDict(),
            )
            for segment in meta[k]:
                if not segment["label"] in item["segments"]:
                    item["segments"][segment["label"]] = []
                item["segments"][segment["label"]].append(
                    (segment["start"] / 1000, segment["end"] / 1000)
                )
            data[k] = item
        fold_datas.append(data)

    test_meta, test_data = fold_metas[test_id], fold_datas[test_id]
    valid_meta, valid_data = fold_metas[valid_id], fold_datas[valid_id]
    train_meta, train_data = {}, {}
    for idx in train_ids:
        train_meta = {**train_meta, **fold_metas[idx]}
        train_data = {**train_data, **fold_datas[idx]}

    def get_labels(meta):
        all_labels = []
        for key in meta:
            for segment in meta[key]:
                all_labels.append(segment["label"])
        return all_labels

    all_classes = list(
        set(get_labels(train_meta) + get_labels(valid_meta) + get_labels(test_meta))
    )
    category = CategoryEncoder(all_classes)

    return Container(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        valid_target_events=valid_meta,
        test_target_events=test_meta,
        category=category,
    )


def hear_scene_trainvaltest(dataset_root: str):
    dataset_root = Path(dataset_root)
    wav_root = dataset_root / "16000"

    def load_json(filepath):
        with open(filepath, "r") as fp:
            return json.load(fp)

    def split_to_data(split: str):
        meta = load_json(dataset_root / f"{split}.json")
        data = {}
        for k in list(meta.keys()):
            wav_path = wav_root / split / k
            labels = meta[k]
            item = dict(
                wav_path=wav_path,
                labels=labels,
            )
            data[k] = item
        return data

    train_data = split_to_data("train")
    valid_data = split_to_data("valid")
    test_data = split_to_data("test")

    def get_labels(data):
        labels = []
        for v in data.values():
            labels.extend(v["labels"])
        return list(set(labels))

    all_labels = list(
        set(get_labels(test_data) + get_labels(valid_data) + get_labels(train_data))
    )
    category = CategoryEncoder(all_labels)

    return Container(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        category=category,
    )


def hear_scene_kfolds(dataset_root: str, test_fold: int = 0, num_folds: int = 5):
    dataset_root = Path(dataset_root)
    wav_root = dataset_root / "16000"

    def load_json(filepath):
        with open(filepath, "r") as fp:
            return json.load(fp)

    test_id = test_fold
    valid_id = (test_fold + 1) % num_folds
    train_ids = [idx for idx in range(num_folds) if idx not in [test_id, valid_id]]

    fold_metas = []
    fold_datas = []
    for fold_id in range(num_folds):
        meta = load_json(dataset_root / f"fold{fold_id:2d}.json".replace(" ", "0"))
        fold_metas.append(meta)

        data = {}
        for k in list(meta.keys()):
            wav_path = wav_root / f"fold{fold_id:2d}".replace(" ", "0") / k
            labels = meta[k]
            assert len(labels) == 1
            item = dict(
                wav_path=wav_path,
                labels=labels,
            )
            data[k] = item
        fold_datas.append(data)

    test_data = fold_datas[test_id]
    valid_data = fold_datas[valid_id]
    train_data = {}
    for idx in train_ids:
        train_data = {**train_data, **fold_datas[idx]}

    def get_labels(data):
        labels = []
        for v in data.values():
            labels.extend(v["labels"])
        return list(set(labels))

    all_labels = list(
        set(get_labels(test_data) + get_labels(valid_data) + get_labels(train_data))
    )
    category = CategoryEncoder(all_labels)

    return Container(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        category=category,
    )
