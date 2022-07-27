import json
from hjson import OrderedDict
import pandas as pd
from pathlib import Path
import torchaudio
from s3prl import Container


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
