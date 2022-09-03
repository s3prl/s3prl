import logging
import os
from collections import defaultdict
from pathlib import Path

from filelock import FileLock
from joblib import Parallel, delayed
from tqdm import tqdm

from s3prl import Container, Output, cache
from s3prl.base.cache import get_cache_root
from s3prl.util.download import _urls_to_filepaths

from .base import Corpus

SPLIT_FILE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
TRIAL_FILE_URL = "https://openslr.magicdatatech.com/resources/49/voxceleb1_test_v2.txt"


class VoxCeleb1SV(Corpus):
    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        train_path, valid_path, test_path, speakerid2label = self.format_path(
            self.dataset_root
        )
        self.categories = speakerid2label
        self.train_data = self.path2data(train_path, speakerid2label)
        self.valid_data = self.path2data(valid_path, speakerid2label)
        self.test_data = {
            self.path2uid(path): {"wav_path": path, "label": None} for path in test_path
        }
        self.test_trials = self.format_test_trials()

    @classmethod
    def path2uid(cls, path):
        return "-".join(Path(path).parts[-3:])

    @classmethod
    def path2data(cls, paths, speakerid2label):
        data = {
            cls.path2uid(path): {
                "wav_path": path,
                "label": speakerid2label[Path(path).parts[-3]],
            }
            for path in paths
        }
        return data

    @staticmethod
    @cache()
    def format_path(dataset_root):
        split_filename = SPLIT_FILE_URL.split("/")[-1]
        split_filepath = get_cache_root() / split_filename
        if not split_filepath.is_file():
            with FileLock(str(split_filepath) + ".lock"):
                os.system(f"wget {SPLIT_FILE_URL} -O {str(split_filepath)}")

        usage_list = open(split_filepath, "r").readlines()
        train, valid, test = [], [], []
        test_list = [
            item
            for item in usage_list
            if int(item.split(" ")[1].split("/")[0][2:]) in range(10270, 10310)
        ]
        usage_list = list(set(usage_list).difference(set(test_list)))
        test_list = [item.split(" ")[1] for item in test_list]

        logging.info("search specified wav name for each split")
        speakerids = []

        for string in tqdm(usage_list, desc="Search train, dev wavs"):
            pair = string.split()
            index = pair[0]
            x = list(dataset_root.glob("dev/wav/" + pair[1]))
            speakerStr = pair[1].split("/")[0]
            if speakerStr not in speakerids:
                speakerids.append(speakerStr)
            if int(index) == 1 or int(index) == 3:
                train.append(str(x[0]))
            elif int(index) == 2:
                valid.append(str(x[0]))
            else:
                raise ValueError

        speakerids = sorted(speakerids)
        speakerid2label = {}
        for idx, spk in enumerate(speakerids):
            speakerid2label[spk] = idx

        for string in tqdm(test_list, desc="Search test wavs"):
            x = list(dataset_root.glob("test/wav/" + string.strip()))
            test.append(str(x[0]))
        logging.info(
            f"finish searching wav: train {len(train)}; valid {len(valid)}; test {len(test)} files found"
        )

        return train, valid, test, speakerid2label

    @classmethod
    def format_test_trials(cls):
        trial_filename = TRIAL_FILE_URL.split("/")[-1]
        trial_filepath = get_cache_root() / trial_filename
        if not trial_filepath.is_file():
            with FileLock(str(trial_filepath) + ".lock"):
                os.system(f"wget {TRIAL_FILE_URL} -O {str(trial_filepath)}")

        trial_list = open(trial_filepath, "r").readlines()
        test_trials = []
        for string in tqdm(trial_list, desc="Prepare testing trials"):
            pair = string.split()
            test_trials.append(
                (int(pair[0]), cls.path2uid(pair[1]), cls.path2uid(pair[2]))
            )

        return test_trials

    @property
    def all_data(self):
        return self.train_data, self.valid_data, self.test_data, self.test_trial

    @property
    def data_split_ids(self):
        return None


def voxceleb1_for_sv(dataset_root: str, n_jobs: int = 4):
    corpus = VoxCeleb1SV(dataset_root, n_jobs)
    all_data = Container(corpus.train_data).add(corpus.valid_data)

    ignored_utts_path = _urls_to_filepaths(
        "https://huggingface.co/datasets/s3prl/voxceleb1_too_short_utts/raw/main/utt"
    )
    with open(ignored_utts_path) as file:
        ignored_utts = [line.strip() for line in file.readlines()]

    for utt in ignored_utts:
        assert utt in all_data

    all_data = Container({k: v for k, v in all_data.items() if k not in ignored_utts})
    return Output(
        train_data=all_data,
        valid_data=corpus.valid_data,
        test_data=corpus.test_data,
        trials=corpus.test_trials,
        categories=corpus.categories,
    )