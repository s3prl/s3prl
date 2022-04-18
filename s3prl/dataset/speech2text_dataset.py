from typing import List
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
from torch.nn.utils.rnn import pad_sequence

from s3prl import Output, cache
from s3prl.util.loader import Loader, TorchaudioLoader, TorchaudioMetadataLoader
from s3prl.util.tokenizer import Tokenizer
from .base import Dataset, in_metadata_mode


class Speech2TextDataset(Dataset):
    def __init__(
        self,
        source: List[Path],
        label: List[str],
        source_loader: Loader = None,
        label_loader: Tokenizer = None,
        metadata_loader: Loader = None,
        metadata_jobs: int = 8,
        name: List[str] = None,
    ) -> None:
        """Speech2TextDataset

        Args:
            source:
                list of sources (paths) of the input
                e.g. [Path("path1"), Path("path2"), ...]
            label:
                list of labels, the order should be sync with sources
                e.g. ["this is a dog", "hello world", ...]
            source_loader:
                Loader, source_loader(sources[0]) to get a actual **input**
                **input** (torch.Tensor): input.dim() == 2, (timestamps, hidden_size)
                    If the input is a waveform, (timestamps, 1)
            label_loader:
                Loader, label_loader(labels[0]) to get a actual **output**
                **output** (torch.Tensor): output.dim() == 1, (textlength, )
                    If the output is a text sequence, (textlength, )
        """
        super().__init__()
        self.name = name or source
        self.sources = source
        self.source_loader = source_loader or TorchaudioLoader()
        self.label_loader = label_loader

        assert len(source) == len(label)

        # prepare metadata
        self.metadata_loader = metadata_loader or TorchaudioMetadataLoader()
        self.metadata_jobs = metadata_jobs
        self.metadatas = self.read_metadata(self.sources)

        # tokenize targets
        self.labels = []
        for i, lab in enumerate(tqdm(label, desc="Tokenizing labels")):
            self.labels.append(self.label_loader.encode(lab))

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
        return Output(x=x, label=label, name=self.name[index])

    def __len__(self):
        return len(self.sources)

    def collate_fn(self, samples) -> Output:
        xs, labels, names = [], [], []
        for sample in samples:
            xs.append(sample.x)
            labels.append(torch.LongTensor(sample.label))
            names.append(sample.name)
        xs_len = torch.LongTensor([len(x) for x in xs])
        xs = pad_sequence(xs, batch_first=True)
        return Output(x=xs, x_len=xs_len, label=labels, name=names)

    def statistics(self):
        return Output()
