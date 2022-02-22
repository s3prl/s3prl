from typing import List
from pathlib import Path

from torch.nn.utils.rnn import pad_sequence

from s3prl import init, Output
from .base import Dataset
from s3prl.util import Loader


class UtteranceClassificationDataset(Dataset):
    """
    The input argument should be easy for the users to replace with their own data
    That is, the data format of the inputs should be intuitive and relate to the task definition
    The Datasets in S3PRL are designed to convert the intuitive input data format into more
    sophisticated format for the high-performance training pipeline: (padding masks, stft masks)
    Hence, the dataset specific operations should be done only in Preprocessors.
    """

    @init.method
    def __init__(
        self,
        sources: List[Path],
        labels: List[str],
        categories: List[str],
        source_loader: Loader = None,
    ) -> None:
        """
        Args:
            sources: list of sources (paths) of the wavforms
            utterance_source_loader:
                Loader, loader.load(utterance_sources[0]) to get a actual wavform
                which is in torch.Tensor and dim() == 1 (single channel)
            labels: list of labels, the order should be sync with utterance_sources
            categories:
                list of strings. all the possible classes. should be the super set for utterance_labels
        """
        super().__init__()

    def __getitem__(self, index):
        path = self.arguments.utterance_sources[index]
        data = self.arguments.utterance_source_loader.load(path)
        label = self.arguments.categories.index(self.arguments.utterance_labels[index])
        return data, label

    def prepare_data(self):
        return self.arguments.categories

    def collate_fn(self, samples):
        datas, labels = [], [], []
        for data, label in samples:
            datas.append(data)
            labels.append(label)
        wavs_len = [len(wav) for wav in wavs]
        wavs = pad_sequence(datas, batch_first=True)
        return Output(wav=wavs, wav_len=wavs_len)
