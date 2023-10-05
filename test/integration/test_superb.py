import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from s3prl.problem import (
    SuperbASR,
    SuperbASV,
    SuperbER,
    SuperbIC,
    SuperbKS,
    SuperbPR,
    SuperbSD,
    SuperbSF,
    SuperbSID,
)
from s3prl.util.pseudo_data import pseudo_audio


@pytest.mark.parametrize("vocab_type", ["subword", "character"])
def test_superb_asr(vocab_type):
    if vocab_type == "subword":
        vocab_args = {"vocab_size": 18}
    else:
        vocab_args = {}

    with tempfile.TemporaryDirectory() as tempdir:
        with pseudo_audio([10, 2, 1, 8, 5]) as (wav_paths, num_samples):

            class TestASR(SuperbASR):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only: bool = False,
                ):
                    all_wav_paths = wav_paths
                    all_text = [
                        "hello how are you today",
                        "fine",
                        "oh",
                        "I think is good",
                        "maybe okay",
                    ]

                    ids = list(range(len(all_wav_paths)))
                    df = pd.DataFrame(
                        data={
                            "id": ids,
                            "wav_path": all_wav_paths,
                            "transcription": all_text,
                        }
                    )
                    train_path = Path(target_dir) / "train.csv"
                    valid_path = Path(target_dir) / "valid.csv"
                    test_path = Path(target_dir) / "test.csv"

                    df.iloc[:3].to_csv(train_path, index=False)
                    df.iloc[3:4].to_csv(valid_path, index=False)
                    df.iloc[4:].to_csv(test_path, index=False)

                    return train_path, valid_path, [test_path]

            problem = TestASR()
            config = problem.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = 2
            config["train"]["save_step"] = 2
            config["eval_batch"] = 2
            config["build_tokenizer"] = {
                "vocab_type": vocab_type,
                "vocab_args": vocab_args,
            }
            config["build_upstream"]["name"] = "fbank"
            problem.run(**config)


def test_superb_er():
    with tempfile.TemporaryDirectory() as tempdir:
        with pseudo_audio([10, 2, 1, 8, 5]) as (wav_paths, num_samples):

            class TestER(SuperbER):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only: bool = False,
                ):
                    ids = [Path(path).stem for path in wav_paths]
                    labels = ["a", "b", "a", "c", "d"]
                    start_secs = [0.0, 0.1, 0.2, None, 0.0]
                    end_secs = [5.2, 1.0, 0.3, None, 4.9]
                    df = pd.DataFrame(
                        data={
                            "id": ids,
                            "wav_path": wav_paths,
                            "label": labels,
                            "start_sec": start_secs,
                            "end_sec": end_secs,
                        }
                    )
                    train_csv = target_dir / "train.csv"
                    valid_csv = target_dir / "valid.csv"
                    test_csv = target_dir / "test.csv"

                    df.to_csv(train_csv)
                    df.to_csv(valid_csv)
                    df.to_csv(test_csv)

                    return train_csv, valid_csv, [test_csv]

            problem = TestER()
            config = problem.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = 2
            config["train"]["save_step"] = 2
            config["eval_batch"] = 2
            config["build_upstream"]["name"] = "fbank"
            problem.run(**config)


def test_superb_ks():
    with tempfile.TemporaryDirectory() as tempdir:
        with pseudo_audio([10, 2, 1, 8, 5]) as (wav_paths, num_samples):

            class TestKS(SuperbKS):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only: bool = False,
                ):
                    ids = [Path(path).stem for path in wav_paths]
                    labels = ["a", "b", "a", "c", "d"]
                    start_secs = [0.0, 0.1, 0.2, None, 0.0]
                    end_secs = [5.2, 1.0, 0.3, None, 4.9]
                    df = pd.DataFrame(
                        data={
                            "id": ids,
                            "wav_path": wav_paths,
                            "label": labels,
                            "start_sec": start_secs,
                            "end_sec": end_secs,
                        }
                    )
                    train_csv = target_dir / "train.csv"
                    valid_csv = target_dir / "valid.csv"
                    test_csv = target_dir / "test.csv"

                    df.to_csv(train_csv)
                    df.to_csv(valid_csv)
                    df.to_csv(test_csv)

                    return train_csv, valid_csv, [test_csv]

            problem = TestKS()
            config = problem.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = 2
            config["train"]["save_step"] = 2
            config["eval_batch"] = 2
            config["build_upstream"]["name"] = "fbank"
            problem.run(**config)


def test_superb_pr():
    with tempfile.TemporaryDirectory() as tempdir:
        with pseudo_audio([10, 2, 1, 8, 5]) as (wav_paths, num_samples):

            class TestPR(SuperbPR):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only: bool = False,
                ):
                    from s3prl.dataio.encoder.g2p import G2P

                    all_wav_paths = wav_paths
                    all_text = [
                        "hello how are you today",
                        "fine",
                        "oh",
                        "I think is good",
                        "maybe okay",
                    ]

                    g2p = G2P()
                    all_text = [g2p.encode(text.strip()) for text in all_text]

                    ids = list(range(len(all_wav_paths)))
                    df = pd.DataFrame(
                        data={
                            "id": ids,
                            "wav_path": all_wav_paths,
                            "transcription": all_text,
                        }
                    )
                    train_path = Path(target_dir) / "train.csv"
                    valid_path = Path(target_dir) / "valid.csv"
                    test_path = Path(target_dir) / "test.csv"

                    df.iloc[:3].to_csv(train_path, index=False)
                    df.iloc[3:4].to_csv(valid_path, index=False)
                    df.iloc[4:].to_csv(test_path, index=False)

                    return train_path, valid_path, [test_path]

            problem = TestPR()
            config = problem.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = 2
            config["train"]["save_step"] = 2
            config["eval_batch"] = 2
            config["build_upstream"]["name"] = "fbank"
            problem.run(**config)


def test_superb_ic():
    with tempfile.TemporaryDirectory() as tempdir:
        with pseudo_audio([10, 2, 1, 8, 5]) as (wav_paths, num_samples):

            class TestIC(SuperbIC):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only: bool = False,
                ):
                    ids = [Path(path).stem for path in wav_paths]
                    labels1 = ["a", "b", "a", "c", "d"]
                    labels2 = ["1", "2", "3", "4", "5"]
                    df = pd.DataFrame(
                        data={
                            "id": ids,
                            "wav_path": wav_paths,
                            "labels": [
                                f"{label1} ; {label2}"
                                for label1, label2 in zip(labels1, labels2)
                            ],
                        }
                    )
                    train_csv = target_dir / "train.csv"
                    valid_csv = target_dir / "valid.csv"
                    test_csv = target_dir / "test.csv"

                    df.to_csv(train_csv)
                    df.to_csv(valid_csv)
                    df.to_csv(test_csv)

                    return train_csv, valid_csv, [test_csv]

            problem = TestIC()
            config = problem.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = 2
            config["train"]["save_step"] = 2
            config["eval_batch"] = 2
            config["build_upstream"]["name"] = "fbank"
            problem.run(**config)


def test_superb_sid():
    with tempfile.TemporaryDirectory() as tempdir:
        with pseudo_audio([10, 2, 1, 8, 5]) as (wav_paths, num_samples):

            class TestSID(SuperbSID):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only: bool = False,
                ):
                    ids = [Path(path).stem for path in wav_paths]
                    label = ["a", "b", "a", "c", "d"]
                    start_secs = [0.0, 0.1, 0.2, None, 0.0]
                    end_secs = [5.2, 1.0, 0.3, None, 4.9]
                    df = pd.DataFrame(
                        data={
                            "id": ids,
                            "wav_path": wav_paths,
                            "label": label,
                            "start_sec": start_secs,
                            "end_sec": end_secs,
                        }
                    )
                    train_csv = target_dir / "train.csv"
                    valid_csv = target_dir / "valid.csv"
                    test_csv = target_dir / "test.csv"

                    df.to_csv(train_csv)
                    df.to_csv(valid_csv)
                    df.to_csv(test_csv)

                    return train_csv, valid_csv, [test_csv]

            problem = TestSID()
            config = problem.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = 2
            config["train"]["save_step"] = 2
            config["eval_batch"] = 2
            config["build_upstream"]["name"] = "fbank"
            problem.run(**config)


def test_superb_sd():
    with tempfile.TemporaryDirectory() as tempdir:
        secs = [10, 2, 1, 8, 5]
        with pseudo_audio(secs) as (wav_paths, num_samples):

            class TestSD(SuperbSD):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only=False,
                ):
                    record_id = [Path(path).stem for path in wav_paths]
                    durations = secs
                    speaker = ["a", "b", "a", "a", "b"]
                    utt_id = record_id
                    start_secs = [0.0, 0.1, 0.2, 0.3, 0.0]
                    end_secs = [5.2, 1.0, 0.3, 5.4, 4.9]
                    df = pd.DataFrame(
                        data={
                            "record_id": record_id,
                            "wav_path": wav_paths,
                            "duration": durations,
                            "utt_id": utt_id,
                            "speaker": speaker,
                            "start_sec": start_secs,
                            "end_sec": end_secs,
                        }
                    )

                    train_csv = Path(target_dir) / "train.csv"
                    valid_csv = Path(target_dir) / "valid.csv"
                    test_csv = Path(target_dir) / "test.csv"

                    df.to_csv(train_csv)
                    df.to_csv(valid_csv)
                    df.to_csv(test_csv)

                    return train_csv, valid_csv, [test_csv]

            problem = TestSD()
            config = problem.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = 2
            config["train"]["save_step"] = 2
            config["eval_batch"] = 2
            config["build_upstream"]["name"] = "fbank"
            problem.run(**config)


def test_superb_asv():
    with tempfile.TemporaryDirectory() as tempdir:
        secs = [10, 2, 1, 8, 5]
        with pseudo_audio(secs) as (wav_paths, num_samples):

            class TestASV(SuperbASV):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only: bool = False,
                ):
                    train_csv = Path(target_dir) / "train.csv"
                    test_csv = Path(target_dir) / "test.csv"

                    ids = [Path(path).stem for path in wav_paths]
                    spk = ["a", "b", "c", "a", "b"]
                    train_df = pd.DataFrame(
                        data={
                            "id": ids,
                            "wav_path": wav_paths,
                            "spk": spk,
                        }
                    )
                    train_df.to_csv(train_csv)

                    id1 = [ids[0], ids[1], ids[2]]
                    id2 = [ids[1], ids[1], ids[2]]
                    wav_path1 = [wav_paths[0], wav_paths[1], wav_paths[2]]
                    wav_path2 = [wav_paths[1], wav_paths[1], wav_paths[2]]
                    labels = [0, 1, 1]
                    test_df = pd.DataFrame(
                        data={
                            "id1": id1,
                            "id2": id2,
                            "wav_path1": wav_path1,
                            "wav_path2": wav_path2,
                            "label": labels,
                        }
                    )
                    test_df.to_csv(test_csv)

                    return train_csv, [test_csv]

            problem = TestASV()
            config = problem.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = math.inf
            config["train"]["save_step"] = 1
            config["build_upstream"]["name"] = "fbank"
            problem.run(**config)


@pytest.mark.parametrize("vocab_type", ["subword", "character"])
def test_superb_sf(vocab_type):
    if vocab_type == "subword":
        vocab_args = {"vocab_size": 22}
    else:
        vocab_args = {}

    with tempfile.TemporaryDirectory() as tempdir:
        with pseudo_audio([10, 2, 1, 8, 5]) as (wav_paths, num_samples):

            class TestSF(SuperbSF):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only: bool = False,
                ):
                    all_wav_paths = wav_paths
                    all_text_with_iob = [
                        ("hello how are you today", "O O O O timeRange"),
                        ("fine thank you", "condition O O"),
                        ("oh nice", "O condition"),
                        ("I think is good", "O O O genre"),
                        ("maybe okay", "O genre"),
                    ]
                    text, iob = zip(*all_text_with_iob)

                    ids = list(range(len(all_wav_paths)))
                    df = pd.DataFrame(
                        data={
                            "id": ids,
                            "wav_path": all_wav_paths,
                            "transcription": text,
                            "iob": iob,
                        }
                    )
                    train_path = Path(target_dir) / "train.csv"
                    valid_path = Path(target_dir) / "valid.csv"
                    test_path = Path(target_dir) / "test.csv"

                    df.iloc[:3].to_csv(train_path, index=False)
                    df.iloc[3:4].to_csv(valid_path, index=False)
                    df.iloc[4:].to_csv(test_path, index=False)

                    return train_path, valid_path, [test_path]

            problem = TestSF()
            config = problem.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = 2
            config["train"]["save_step"] = 2
            config["eval_batch"] = 2
            config["build_tokenizer"] = {
                "vocab_type": vocab_type,
                "vocab_args": vocab_args,
            }
            config["build_upstream"]["name"] = "fbank"
            problem.run(**config)
