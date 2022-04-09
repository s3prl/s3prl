from typing import List
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.nn.functional as F
import numpy as np
import random

from s3prl import Output, cache
from s3prl.util.loader import Loader, TorchaudioLoader, TorchaudioMetadataLoader
from .base import Dataset, in_metadata_mode

def pad_audio(waveform, sample_rate=16000, second=-1):
    """
    Args:
        waveform: 
            (torch.Tensor): input.dim() == 2, (timestamps, 1)
        second:
            int, indicating the returned audio duration
            If second = -1, return the entire waveform
    """

    audio_length = waveform.shape[0]

    if second <= 0:
        return waveform

    length = np.int64(sample_rate * second)

    if audio_length <= length:
        shortage = length - audio_length
        waveform = F.pad(waveform.view(1, 1, -1), (0, shortage), "circular")
        waveform = waveform.view(-1, 1)
    else:
        start = np.int64(random.random() * (audio_length - length))
        waveform =  waveform[start:start+length, :]
    return waveform


class SpeakerClassificationDataset(Dataset):
    """
    The input argument should be easy for the users to replace with their own data
    That is, the data format of the inputs should be intuitive and relate to the task definition
    The Datasets in S3PRL are designed to convert the intuitive input data format into more
    sophisticated format for the high-performance training pipeline: (padding masks, stft masks)
    Hence, the dataset specific operations should be done only in Preprocessors.
    """

    def __init__(
        self,
        source: List[Path],
        label: List[int],
        category: dict,
        source_loader: Loader = None,
        metadata_loader: Loader = None,
        metadata_jobs: int = 8,
        name: List[str] = None,
    ) -> None:
        """
        Args:
            source:
                list of sources (paths) of the input
                e.g. ["path1", "path2", ...]
            label:
                list of labels, the order should be sync with sources
                e.g. [1, 1024, 512 ...]
            category:
                dictionary with speakerid-label as key-value pairs
                e.g. {'id10001':0, 'id10200':199 ...}
            source_loader:
                Loader, source_loader.load(sources[0]) to get a actual **input**
                **input** (torch.Tensor): input.dim() == 2, (timestamps, 1)
        """
        super().__init__()
        self.sources = source
        self.labels = label
        self.categories = category
        self.source_loader = source_loader or TorchaudioLoader()

        # prepare metadata
        self.metadata_loader = metadata_loader or TorchaudioMetadataLoader()
        self.metadata_jobs = metadata_jobs
        self.metadatas = self.read_metadata(self.sources)

    @cache(signatures=["sources"])
    def read_metadata(self, sources):
        metadatas = Parallel(n_jobs=self.metadata_jobs)(
            delayed(self.metadata_loader)(source)
            for source in tqdm(sources, desc="Reading metadata")
        )
        return metadatas
    
    def __getitem__(self, index):
        if in_metadata_mode():
            return Output(timestamp=self.metadatas[index].timestamp)

        path = Path(self.sources[index])
        x = self.source_loader(path).output
        label = self.labels[index]
        return Output(x=x, label=label, name="/".join(path.resolve().parts[-3:]))

    def __len__(self):
        return len(self.sources)

    def collate_fn(self, samples):
        xs, labels, names = [], [], []
        for sample in samples:
            xs.append(pad_audio(sample.x, sample_rate=self.source_loader.sample_rate, second=3))
            labels.append(sample.label)
            names.append(sample.name)
        xs_len = torch.LongTensor([len(x) for x in xs])
        xs = torch.cat(xs, dim=1).transpose(1, 0).unsqueeze(-1)
        return Output(x=xs, x_len=xs_len, label=labels, name=names)

    def statistics(self):
        return Output()

class SpeakerTrialDataset(Dataset):
    """
    The input argument should be easy for the users to replace with their own data
    That is, the data format of the inputs should be intuitive and relate to the task definition
    The Datasets in S3PRL are designed to convert the intuitive input data format into more
    sophisticated format for the high-performance training pipeline: (padding masks, stft masks)
    Hence, the dataset specific operations should be done only in Preprocessors.
    """

    def __init__(
        self,
        source: List[str],
        label: List[tuple],
        source_loader: Loader = None,
        metadata_loader: Loader = None,
        metadata_jobs: int = 8,
        name: List[str] = None,
    ) -> None:
        """
        Args:
            source:
                list of sources (paths) of the input
                e.g. ["path1", "path2", ...]
            label:
                list of tuples (enroll_path, test_path, label), the order should be sync with sources
                e.g. [(enroll_path1, test_path1, 1), (enroll_path2, test_path2, 0) ...]
            source_loader:
                Loader, source_loader.load(sources[0]) to get a actual **input**
                **input** (torch.Tensor): input.dim() == 2, (timestamps, 1)
        """
        super().__init__()
        self.sources = source
        self.trials = label
        self.source_loader = source_loader or TorchaudioLoader()

        # prepare metadata
        self.metadata_loader = metadata_loader or TorchaudioMetadataLoader()
        self.metadata_jobs = metadata_jobs
        self.metadatas = self.read_metadata(self.sources)

    @cache(signatures=["sources"])
    def read_metadata(self, sources):
        metadatas = Parallel(n_jobs=self.metadata_jobs)(
            delayed(self.metadata_loader)(source)
            for source in tqdm(sources, desc="Reading metadata")
        )
        return metadatas

    def __getitem__(self, index):
        if in_metadata_mode():
            return Output(timestamp=self.metadatas[index].timestamp)

        path = Path(self.sources[index])
        x = self.source_loader(path).output
        return Output(x=x, name=self.sources[index])

    def __len__(self):
        return len(self.sources)

    def collate_fn(self, samples):
        xs, names = [], []
        for sample in samples:
            xs.append(pad_audio(sample.x, sample_rate=self.source_loader.sample_rate, second=-1))
            names.append(sample.name)
        xs_len = torch.LongTensor([len(x) for x in xs])
        xs = torch.cat(xs, dim=1).transpose(1, 0).unsqueeze(-1)
        return Output(x=xs, x_len=xs_len, name=names)

    def statistics(self):
        return Output(label=self.trials)