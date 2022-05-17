import os
from collections import defaultdict
import logging
from pathlib import Path

from filelock import FileLock
from joblib import Parallel, delayed
from tqdm import tqdm

from s3prl import Output, cache

from .base import Corpus


SPLIT_FILE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
TRIAL_FILE_URL = "https://openslr.magicdatatech.com/resources/49/voxceleb1_test_v2.txt"

class VoxCeleb1(Corpus):
    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        
        self.dataset_root = Path(dataset_root).resolve()
        
        train_path, valid_path, test_path, speakerid2label = self.format_path(self.dataset_root)
        self.categories = speakerid2label
        self.train_data = self.path2data(train_path, speakerid2label)
        self.valid_data = self.path2data(valid_path, speakerid2label)
        self.test_data  = {uid: {"wav_path": uid, "label": None} for uid in test_path}
        
        self.test_trials = self.format_test_trials(self.dataset_root)

    @staticmethod
    def path2data(path, speakerid2label):
        data = {
            uid: {
                "wav_path": uid,
                "label": speakerid2label[uid.split("/")[-3]],
            }
            for uid in path
        }
        return data

    @staticmethod
    @cache()
    def format_path(dataset_root):

        split_filename = SPLIT_FILE_URL.split("/")[-1]
        split_filepath = dataset_root / split_filename
        if not split_filepath.is_file():
            with FileLock(str(split_filepath) + ".lock"):
                os.system(f"wget {SPLIT_FILE_URL} -O {str(split_filepath)}")

        usage_list = open(split_filepath, "r").readlines()
        train, valid, test = [], [], []
        test_list  = [item for item in usage_list if int(item.split(' ')[1].split('/')[0][2:]) in range(10270, 10310)]
        usage_list = list(set(usage_list).difference(set(test_list)))
        test_list  = [item.split(" ")[1] for item in test_list]

        logging.info("search specified wav name for each split")
        speakerids = []
        
        for string in tqdm(usage_list, desc="Search train, dev wavs"):
            pair  = string.split()
            index = pair[0]
            x     = list(dataset_root.glob("dev/wav/" + pair[1]))
            speakerStr = pair[1].split('/')[0]
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

    @staticmethod
    @cache()
    def format_test_trials(dataset_root):

        trial_filename = TRIAL_FILE_URL.split("/")[-1]
        trial_filepath = dataset_root / trial_filename
        if not trial_filepath.is_file():
            with FileLock(str(trial_filepath) + ".lock"):
                os.system(f"wget {TRIAL_FILE_URL} -O {str(trial_filepath)}")
        
        usage_list  = open(trial_filepath, "r").readlines()

        test_trials = []
        prefix      = dataset_root / "test/wav"

        for string in tqdm(usage_list, desc="Prepare testing trials"):
            pair = string.split()
            test_trials.append((int(pair[0]), str(prefix / pair[1]), str(prefix / pair[2])))

        return test_trials

    @property
    def all_data(self):
        return self.train_data, self.valid_data, self.test_data, self.test_trial

    @property
    def data_split_ids(self):
        return None


class VoxCeleb1SV(VoxCeleb1):
    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        super().__init__(dataset_root, n_jobs)

    def __call__(self):
        return Output(
            train_data=self.train_data,
            valid_data=self.valid_data,
            test_data=self.test_data,
            trials=self.test_trials,
            categories=self.categories
        )
