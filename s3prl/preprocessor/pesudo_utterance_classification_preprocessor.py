import random
from unicodedata import name
from librosa.util import find_files
import torchaudio

from s3prl import Object, init, Output
from .util import cache

PSEUDO_ALL_LABELS = ["A", "B", "C"]

class PseudoUtteranceClassificationPreprocessor(Object):
    @init.method
    def __init__(self, dataset_root, train_num=3, dev_num=1, test_num=1):
        super().__init__()
        self.dataset_root = dataset_root

        # TODO: This should be in the real case
        #stage1
        self.paths = self.all_paths(dataset_root)
        self.paths = ["1", "2", "3", "4", "5"]

        #stage2
        self.train, self.dev, self.test = self.standard_splits(self.paths)

        #stage3
        self.train_label = self.path2labels(self.train)
        self.dev_label = self.path2labels(self.dev)
        self.test_label = self.path2labels(self.test)
        self.categories = self.get_categories(
            [*self.train_label, *self.dev_label, *self.test_label]
        )

        for path in self.train:
            fbank = self.fbank(path)

    @cache
    @staticmethod
    def fbank(path):
        wav, sr = torchaudio.load(path)
        fbank = torchaudio.compliance.kaldi.fbank(wav)
        return fbank

    @cache
    @staticmethod
    def all_paths(dataset_root):
        # slow
        all_files = find_files(dataset_root)
        return all_files

    @cache
    @staticmethod
    def path2labels(paths):
        return [random.choice(PSEUDO_ALL_LABELS) for _ in paths]

    @cache
    @staticmethod
    def get_categories(labels):
        return set(labels)

    @cache
    @staticmethod
    def standard_splits(paths, train_num, dev_num, test_num):
        return (
            paths[:train_num],
            paths[train_num : train_num + dev_num],
            paths[train_num + dev_num :],
        )

    def preprocess_train(self):
        return Output(
            sources=self.train,
            labels=self.train_label,
            categories=self.categories,
        )

    def preprocess_dev(self):
        return Output(
            sources=self.dev,
            labels=self.dev_label,
            categories=self.categories,
        )

    def preprocess_test(self):
        return Output(
            sources=self.test,
            labels=self.test_label,
            categories=self.categories,
        )
