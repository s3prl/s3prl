from dataclasses import dataclass
from typing import List


class CategoryEncoder:
    def __init__(self, category: List[str]):
        self.category = list(category)

    def __len__(self):
        return len(self.category)

    def encode(self, label):
        return self.category.index(label)

    def decode(self, index):
        return self.category[index]
