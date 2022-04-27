import logging
import os
from pathlib import Path

from tqdm import tqdm

from s3prl import Object, Output, cache
from s3prl.util.loader import TorchaudioLoader

SPLIT_FILE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"


class VoxCeleb1SIDPreprocessor(Object):
    def __init__(self, dataset_root):
        super().__init__()
        self.speaker_num = 1251

        dataset_root = Path(dataset_root).resolve()

        split_filename = SPLIT_FILE_URL.split("/")[-1]
        if not (dataset_root / split_filename).is_file():
            os.system(f"wget {SPLIT_FILE_URL} -O {str(dataset_root)}/{split_filename}")

        self.train_path, self.valid_path, self.test_path = self.standard_split(
            dataset_root, split_filename
        )
        self.train_label = self.build_label(self.train_path)
        self.valid_label = self.build_label(self.valid_path)
        self.test_label = self.build_label(self.test_path)

        categories = list(set([*self.train_label, *self.valid_label, *self.test_label]))
        categories = sorted([int(c.split("_")[-1]) for c in categories])
        self.categories = [f"speaker_{c}" for c in categories]

        assert len(self.categories) == self.speaker_num

    def build_label(self, train_path_list):
        y = []
        for path in train_path_list:
            id_string = path.split("/")[-3]
            y.append(f"speaker_{int(id_string[2:]) - 10001}")
        return y

    @staticmethod
    @cache()
    def standard_split(dataset_root, split_filename):
        meta_data = dataset_root / split_filename
        usage_list = open(meta_data, "r").readlines()

        train, valid, test = [], [], []
        logging.info("search specified wav name for each split")
        for string in tqdm(usage_list, desc="Search wavs"):
            pair = string.split()
            index = pair[0]
            x = list(dataset_root.glob("*/wav/" + pair[1]))
            if int(index) == 1:
                train.append(str(x[0]))
            elif int(index) == 2:
                valid.append(str(x[0]))
            elif int(index) == 3:
                test.append(str(x[0]))
            else:
                raise ValueError
        logging.info(
            f"finish searching wav: train {len(train)}; valid {len(valid)}; test {len(test)} files found"
        )
        return train, valid, test

    def train_data(self):
        return Output(
            source=self.train_path,
            label=self.train_label,
            category=self.categories,
            source_loader=TorchaudioLoader(),
        )

    def valid_data(self):
        return Output(
            source=self.valid_path,
            label=self.valid_label,
            category=self.categories,
            source_loader=TorchaudioLoader(),
        )

    def test_data(self):
        return Output(
            source=self.test_path,
            label=self.test_label,
            category=self.categories,
            source_loader=TorchaudioLoader(),
        )

    def statistics(self):
        return Output(input_size=1, category=self.categories)
