"""
Simple categorical encoder

Authors:
  Leo 2022
"""

from typing import List


class CategoryEncoder:
    def __init__(self, category: List[str]) -> None:
        self.category = list(sorted(set(category)))

    def __len__(self) -> int:
        return len(self.category)

    def encode(self, label: str) -> int:
        return self.category.index(label)

    def decode(self, index: int) -> str:
        return self.category[index]


class CategoryEncoders:
    def __init__(self, categories: List[List[str]]) -> None:
        self.categories = [CategoryEncoder(c) for c in categories]

    def __len__(self) -> int:
        return sum([len(c) for c in self.categories])

    def __iter__(self):
        for category in self.categories:
            yield category

    def encode(self, labels: List[str]) -> List[int]:
        assert len(labels) == len(self.categories)
        return [
            encoder.encode(label) for label, encoder in zip(labels, self.categories)
        ]

    def decode(self, indices: List[int]) -> List[str]:
        return [
            encoder.decode(index) for index, encoder in zip(indices, self.categories)
        ]
