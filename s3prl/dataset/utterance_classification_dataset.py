from pathlib import Path
from typing import List

import torch
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from s3prl import Output, cache
from s3prl.util.loader import Loader, TorchaudioLoader, TorchaudioMetadataLoader

from .base import Dataset, in_metadata_mode


class UtteranceClassificationDataset(Dataset):
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
        label: List[str],
        category: List[str],
        source_loader: Loader = None,
        metadata_loader: Loader = None,
        metadata_jobs: int = 8,
        name: List[str] = None,
    ) -> None:
        """
        Args:
            source:
                list of sources (paths) of the input
                e.g. [Path("path1"), Path("path2"), ...]
            label:
                list of labels, the order should be sync with sources
                e.g. ["happy", "sad", ...]
            category:
                list of strings. all the possible classes. should be the super set for utterance_labels
                e.g. ["happy", "sad", "neutral", "angry"]
            source_loader:
                Loader, source_loader(sources[0]) to get a actual **input**
                **input** (torch.Tensor): input.dim() == 2, (timestamps, hidden_size)
                    If the input is a waveform, (timestamps, 1)
        """
        super().__init__()
        self.name = name or source
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
        label_string = self.labels[index]
        return Output(x=x, label=label_string, name=self.name[index])

    def __len__(self):
        return len(self.sources)

    def collate_fn(self, samples):
        xs, labels, names = [], [], []
        for sample in samples:
            xs.append(sample.x)
            labels.append(sample.label)
            names.append(sample.name)
        xs_len = torch.LongTensor([len(x) for x in xs])
        xs = pad_sequence(xs, batch_first=True)
        return Output(x=xs, x_len=xs_len, label=labels, name=names)

    def statistics(self):
        return Output()
