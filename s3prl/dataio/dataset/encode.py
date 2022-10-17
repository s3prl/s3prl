from typing import List

import torch

from s3prl.dataio.encoder.category import CategoryEncoder, CategoryEncoders
from s3prl.dataio.encoder.tokenizer import Tokenizer

from . import Dataset

__all__ = [
    "EncodeCategory",
    "EncodeCategories",
    "EncodeMultiLabel",
    "EncodeText",
]


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


class EncodeMultiLabel(Dataset):
    def __init__(self, labels: List[List[str]], encoder: CategoryEncoder) -> None:
        super().__init__()
        self.labels = labels
        self.encoder = encoder

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def label_to_binary_vector(label_ids: List[int], num_labels: int) -> torch.Tensor:
        if len(label_ids) == 0:
            binary_labels = torch.zeros((num_labels,), dtype=torch.float)
        else:
            binary_labels = torch.zeros((num_labels,)).scatter(
                0, torch.tensor(label_ids), 1.0
            )

        assert set(torch.where(binary_labels == 1.0)[0].numpy()) == set(label_ids)
        return binary_labels

    def __getitem__(self, index: int):
        labels = self.labels[index]
        label_ids = [self.encoder.encode(label) for label in labels]
        binary_labels = self.label_to_binary_vector(label_ids, len(self.encoder))

        return {
            "labels": labels,
            "binary_labels": binary_labels,
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
