from typing import List

import torch

from s3prl.dataio.encoder.category import CategoryEncoder, CategoryEncoders
from s3prl.dataio.encoder.tokenizer import Tokenizer

from . import Dataset


class EncodeCategory(Dataset):
    def __init__(self, labels: List[str], encoder: CategoryEncoder) -> None:
        super().__init__()
        self.labels = labels
        self.encoder = encoder

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        return {
            "label": label,
            "class_id": self.encoder.encode(label),
        }


class EncodeCategories(Dataset):
    def __init__(self, labels: List[List[str]], encoders: CategoryEncoders) -> None:
        super().__init__()
        self.labels = labels
        self.encoders = encoders

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        labels = self.labels[index]
        return {
            "labels": labels,
            "class_ids": torch.LongTensor(self.encoders.encode(labels)),
        }


class EncodeText(Dataset):
    def __init__(
        self, text: List[str], tokenizer: Tokenizer, iob: List[str] = None
    ) -> None:
        super().__init__()
        self.text = text
        self.iob = iob
        if iob is not None:
            assert len(text) == len(iob)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index: int):
        text = self.text[index]
        if self.iob is not None:
            iob = self.iob[index]
            tokenized_ids = self.tokenizer.encode(text, iob)
            text = self.tokenizer.decode(tokenized_ids)
        else:
            tokenized_ids = self.tokenizer.encode(text)

        return {
            "labels": text,
            "class_ids": torch.LongTensor(tokenized_ids),
        }
