"""
Parse VoxCeleb1 corpus for verification

Authors:
  * Haibin Wu 2022
"""

import logging
from pathlib import Path

from tqdm import tqdm

from s3prl.util.download import _download

from .base import Corpus

SPLIT_FILE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
TRIAL_FILE_URL = "https://openslr.magicdatatech.com/resources/49/voxceleb1_test_v2.txt"

__all__ = [
    "VoxCeleb1SV",
]


class VoxCeleb1SV(Corpus):
    def __init__(
        self, dataset_root: str, download_dir: str, force_download: bool = True
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()

        train_path, valid_path, test_path, speakerid2label = self.format_path(
            self.dataset_root, download_dir, force_download
        )
        self.categories = speakerid2label
        self.train_data = self.path2data(train_path, speakerid2label)
        self.valid_data = self.path2data(valid_path, speakerid2label)
        self.test_data = {
            self.path2uid(path): {"wav_path": path, "label": None} for path in test_path
        }
        self.test_trials = self.format_test_trials(download_dir, force_download)

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
    def format_path(dataset_root, download_dir, force_download: bool):
        split_filename = SPLIT_FILE_URL.split("/")[-1]
        split_filepath = Path(download_dir) / split_filename
        _download(split_filepath, SPLIT_FILE_URL, refresh=force_download)

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
    def format_test_trials(cls, download_dir: str, force_download: bool):
        trial_filename = TRIAL_FILE_URL.split("/")[-1]
        trial_filepath = Path(download_dir) / trial_filename
        _download(trial_filepath, TRIAL_FILE_URL, refresh=force_download)

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
        return self.train_data, self.valid_data, self.test_data, self.test_trials

    @property
    def data_split_ids(self):
        return None
