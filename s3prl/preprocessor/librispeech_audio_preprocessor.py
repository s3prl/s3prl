from librosa.util import find_files

from s3prl.util.loader import TorchaudioLoader
from s3prl import Object, Output, cache


class LibriSpeechAudioPreprocessor(Object):
    def __init__(self, dataset_root, 
                 train_sets = ['train-clean-100', 'train-clean-360', 'train-other-500'],
                 valid_sets = ['dev-clean', 'dev-other'],
                 test_sets = ['test-clean', 'test-other']):
        super().__init__()

        # stage1
        self.train = []
        for s in train_sets:
            self.train += self.all_paths(f"{dataset_root}/{s}")
        
        self.valid = []
        for s in valid_sets:
            self.valid += self.all_paths(f"{dataset_root}/{s}")
        
        self.test = []
        for s in test_sets:
            self.test += self.all_paths(f"{dataset_root}/{s}")

    @staticmethod
    @cache()
    def all_paths(dataset_root):
        all_files = find_files(dataset_root, ext=['aac', 'flac', 'm4a', 'mp3', 'wav'])
        return all_files

    def train_data(self):
        return Output(
            source=self.train,
            source_loader=TorchaudioLoader(),
        )

    def valid_data(self):
        return Output(
            source=self.valid,
            source_loader=TorchaudioLoader(),
        )

    def test_data(self):
        return Output(
            source=self.test,
            source_loader=TorchaudioLoader(),
        )

    def statistics(self):
        return
