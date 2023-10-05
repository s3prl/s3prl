"""
Simple categorical encoder

Authors:
  Shu-wen Yang 2022
"""

from typing import List


class CategoryEncoder:
    def __init__(self, category: List[str]) -> None:
        self.category = list(sorted(set(category)))

    def __len__(self):
        return len(self.category)

    def encode(self, label):
        return self.category.index(label)

    def decode(self, index):
        return self.category[index]


class CategoryEncoders:
    def __init__(self, categories: List[List[str]]) -> None:
        self.categories = [CategoryEncoder(c) for c in categories]

    def __len__(self):
        return sum([len(c) for c in self.categories])

    def __iter__(self) -> List[CategoryEncoder]:
        for c in self.categories:
            yield c
