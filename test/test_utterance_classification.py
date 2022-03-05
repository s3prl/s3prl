import torch

from s3prl import Object, Output
from s3prl.task import UtteranceClassification
from s3prl.dataset import UtteranceClassificationDataset
from s3prl.task.utterance_classification import UtteranceClassifierExample


def test_utterance_classification():
    categories = [1, 2, 3]
    task = UtteranceClassification(UtteranceClassifierExample(3, len(categories)), categories)


class Loader(Object):
    def __init__(self) -> None:
        pass

    def load(self, source):
        return Output(output=torch.randn(16000, 1))


def test_utterance_classification_dataset():

    source = ["a", "b", "c"] * 3
    labels = ["A", "B", "C"] * 3
    categories = sorted(list(set(labels)))
    dataset = UtteranceClassificationDataset(source, labels, categories, Loader())
    output = dataset.collate_fn([dataset[0], dataset[1]])
    wav = output.subset("x")
    assert isinstance(wav, torch.Tensor)
