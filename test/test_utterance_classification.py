import torch

from s3prl import Object, init
from s3prl.task import UtteranceClassification
from s3prl.dataset import UtteranceClassificationDataset
from s3prl.task.utterance_classification import UtteranceClassifier

def test_utterance_classification():
    categories = [1, 2, 3]
    task = UtteranceClassification(UtteranceClassifier(3, len(categories)), categories)

class Loader(Object):
    @init.method
    def __init__(self) -> None:
        pass

    def load(self, source):
        return torch.randn(160000)

def source_loader(source):
    return torch.randn()

def test_utterance_classification_dataset():
    source = ["a", "b", "c"] * 3
    source=loader = Loader()
    labels = ["A", "B", "C"] * 3
    categories = labels
    UtteranceClassificationDataset(source, source_loader, labels, categories)
