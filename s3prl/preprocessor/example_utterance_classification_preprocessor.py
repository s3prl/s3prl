import random

from librosa.util import find_files

from s3prl import Object, Output, cache, init
from s3prl.util.loader import PseudoLoader

PSEUDO_ALL_LABELS = ["happy", "sad", "angry", "neutral"]


class ExampleUtteranceClassificationPreprocessor(Object):
    def __init__(self, dataset_root, train_ratio: float, valid_ratio: float):
        super().__init__()
        self.dataset_root = dataset_root
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

        # stage1
        self.paths = ["path1", "path2", "path3", "path4", "path5"] * 100

        # stage2
        self.train, self.valid, self.test = self.standard_splits(
            self.paths, self.train_ratio, self.valid_ratio
        )

        # stage3
        self.train_label = self.path2labels(self.train)
        self.valid_label = self.path2labels(self.valid)
        self.test_label = self.path2labels(self.test)
        self.categories = list(
            set([*self.train_label, *self.valid_label, *self.test_label])
        )

    @staticmethod
    @cache()
    def all_paths(dataset_root):
        all_files = find_files(dataset_root)
        return all_files

    @staticmethod
    @cache()
    def path2labels(paths):
        return [random.choice(PSEUDO_ALL_LABELS) for _ in paths]

    @staticmethod
    @cache()
    def standard_splits(paths, train_ratio, valid_ratio):
        paths = paths.copy()
        random.seed(0)
        random.shuffle(paths)
        train_num = round(len(paths) * train_ratio)
        valid_num = round(len(paths) * valid_ratio)
        return (
            paths[:train_num],
            paths[train_num : train_num + valid_num],
            paths[train_num + valid_num :],
        )

    def train_data(self):
        return Output(
            source=self.train,
            label=self.train_label,
            category=self.categories,
            source_loader=PseudoLoader(),
        )

    def valid_data(self):
        return Output(
            source=self.valid,
            label=self.valid_label,
            category=self.categories,
            source_loader=PseudoLoader(),
        )

    def test_data(self):
        return Output(
            source=self.test,
            label=self.test_label,
            category=self.categories,
            source_loader=PseudoLoader(),
        )

    def statistics(self):
        return Output(input_size=1, category=self.categories)
