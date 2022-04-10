from typing import List
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
from torch.nn.utils.rnn import pad_sequence

from s3prl import Output, cache
from s3prl.util.loader import Loader, TorchaudioLoader, TorchaudioMetadataLoader
from .base import Dataset, in_metadata_mode


class Speech2TextDataset(Dataset):
    def __init__(
        self,
        source: List[Path],
        target: List[str],
        vocab: List[str],
        source_loader: Loader = None,
        target_loader: Loader = None,
        metadata_loader: Loader = None,
        metadata_jobs: int = 8,
        name: List[str] = None,
    ) -> None:
        """
        Args:
            source:
                list of sources (paths) of the input
                e.g. [Path("path1"), Path("path2"), ...]
            target:
                list of targets, the order should be sync with sources
                e.g. ["this is a dog", "hello world", ...]
            vocab:
                list of vocabularies. all the possible classes.
                e.g. ["a", "b", "c", "d", ...]
            source_loader:
                Loader, source_loader(sources[0]) to get a actual **input**
                **input** (torch.Tensor): input.dim() == 2, (timestamps, hidden_size)
                    If the input is a waveform, (timestamps, 1)
            target_loader:
                Loader, target_loader(targets[0]) to get a actual **output**
                **output** (torch.Tensor): output.dim() == 1, (textlength, )
                    If the output is a text sequence, (textlength, )
        """
        super().__init__()
        self.name = name or source
        self.sources = source
        self.targets = target
        self.vocab_list = vocab
        self.source_loader = source_loader or TorchaudioLoader()
        self.target_loader = target_loader

        # prepare metadata
        self.metadata_loader = metadata_loader or TorchaudioMetadataLoader()
        self.metadata_jobs = metadata_jobs
        self.metadatas = self.read_metadata(self.sources)

        # tokenize targets
        for i, tgt in enumerate(tqdm(self.targets)):
            self.targets[i] = self.target_loader(tgt, desc="Tokenizing targets")

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
        label = self.targets[index]
        return Output(x=x, label=label, name=self.name[index])

    def __len__(self):
        return len(self.sources)

    def collate_fn(self, samples):
        raise NotImplementedError

    def statistics(self):
        return Output()
