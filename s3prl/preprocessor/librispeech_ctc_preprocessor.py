import os
import logging
from tqdm import tqdm
from pathlib import Path

from s3prl.util.loader import TorchaudioLoader
from s3prl import Object, Output, cache


class LibriSpeechCTCPreprocessor(Object):
    def __init__(self, dataset_root, splits):
        super().__init__()

        dataset_root = Path(dataset_root).resolve()

    @staticmethod
    @cache()
    def standard_split(dataset_root, split_filename):
        meta_data = dataset_root / split_filename
        return

    def train_data(self):
        return

    def valid_data(self):
        return

    def test_data(self):
        return

    def statistics(self):
        return
