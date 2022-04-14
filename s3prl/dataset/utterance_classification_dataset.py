import torch

from typing import List
from dataclasses import dataclass

from s3prl import Output
from .base import DatasetBuilder, AugmentedDynamicItemDataset


@dataclass
class CategoryEncoder:
    category: List[str]

    def __len__(self):
        return len(self.category)

    def encode(self, label):
        return self.category.index(label)

    def decode(self, index):
        return self.category[index]


@dataclass
class UtteranceClassificationDatasetBuilder(DatasetBuilder):
    def prepare_category(self, labels):
        return CategoryEncoder(sorted(list(set(labels))))

    def encode_label(self, category, label):
        return category.encode(label)

    def compute_length(self, tensor):
        return len(tensor)

    def build_train_data(
        self,
        data: dict,
        **kwargs,
    ):
        labels = [item["label"] for item in data.values()]
        category = self.prepare_category(labels)
        dataset = self.build_data(data, category).slice(1)
        return Output(dataset=dataset, category=category, output_size=len(category))

    def build_data(
        self,
        data: dict,
        category: CategoryEncoder,
        **kwargs,
    ):
        """
        Args:
            data (dict)
                id:
                    wav_path: str
                    label: str

            category (callable)
                encode: callable, str -> int
                decode: callable, int -> str

        Return:
            AugmentedDynamicItemDataset, with keys:
                x: torch.Tensor, (time, channel)
                x_len: int
                class_id: int
                unique_name: str
        """
        dataset = AugmentedDynamicItemDataset(data)
        dataset.add_global_stats("category", category)
        dataset.add_dynamic_item_and_metadata(
            self.load_audio, takes="wav_path", provide="wav"
        )
        dataset.add_dynamic_item(self.compute_length, takes="wav", provides="wav_len")
        dataset.add_dynamic_item(
            self.encode_label, takes=["category", "label"], provides="class_id"
        )
        dataset.set_output_keys(
            {
                "x": "wav",
                "x_len": "wav_len",
                "class_id": "class_id",
                "unique_name": "id",
            }
        )
        return Output(dataset=dataset)


@dataclass
class UtteranceMultiClassClassificationDatasetBuilder(
    UtteranceClassificationDatasetBuilder
):
    def build_train_data(
        self,
        data: dict,
        **kwargs,
    ):
        labels = [item["labels"] for item in data.values()]
        label_types = list(zip(*labels))
        categories = [self.prepare_category(label_type) for label_type in label_types]
        dataset = self.build_data(data, categories).slice(1)
        return Output(
            dataset=dataset,
            categories=categories,
            output_size=sum([len(category) for category in categories]),
        )

    def encode_label(self, categories, labels):
        return torch.LongTensor(
            [category.encode(label) for category, label in zip(categories, labels)]
        )

    def build_data(
        self,
        data: dict,
        categories: List[CategoryEncoder],
        **kwargs,
    ):
        """
        Args:
            data (dict)
                id:
                    wav_path: str
                    labels: List[str]

            categories List[Category]
                encode: callable, str -> int
                decode: callable, int -> str

        Return:
            AugmentedDynamicItemDataset, with keys:
                x: torch.Tensor, (time, channel)
                x_len: int
                class_id: int
                labels: List[str]
                unique_name: str
        """
        dataset = AugmentedDynamicItemDataset(data)
        dataset.add_global_stats("categories", categories)
        dataset.add_dynamic_item_and_metadata(
            self.load_audio, takes="wav_path", provide="wav"
        )
        dataset.add_dynamic_item(self.compute_length, takes="wav", provides="wav_len")
        dataset.add_dynamic_item(
            self.encode_label, takes=["categories", "labels"], provides="class_ids"
        )
        dataset.set_output_keys(
            {
                "x": "wav",
                "x_len": "wav_len",
                "class_ids": "class_ids",
                "labels": "labels",
                "unique_name": "id",
            }
        )
        return Output(dataset=dataset)
