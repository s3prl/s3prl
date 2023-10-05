"""
Utterance Classification Tasks

Authors
  * Leo 2022
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from s3prl.dataio.encoder.category import CategoryEncoder, CategoryEncoders
from s3prl.metric import accuracy

from . import Task

logger = logging.getLogger(__name__)

__all__ = [
    "UtteranceClassifierExample",
    "UtteranceClassificationTask",
]


class UtteranceClassifierExample(torch.nn.Module):
    """
    Attributes:
        input_size: int
        output_size: int
    """

    def __init__(self, input_size=3, output_size=4):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    def forward(self, x, x_len):
        """
        Args:
            x (torch.Tensor): (batch_size, timestemps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            output (torch.Tensor): (batch_size, output_size)
        """
        assert x.size(-1) == self.input_size
        output = torch.randn(x.size(0), self.output_size)
        assert output


class UtteranceClassificationTask(Task):
    """
    Attributes:
        input_size (int): defined by model.input_size
        output_size (int): defined by len(categories)
    """

    def __init__(self, model: UtteranceClassifierExample, category: CategoryEncoder):
        """
        model.output_size should match len(categories)

        Args:
            model (UtteranceClassifier)
            category:
                encode: str -> int
                decode: int -> str
                __len__: -> int
        """

        super().__init__()
        self.model = model
        self.category = category
        assert self.model.output_size == len(category)

    def predict(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            logits (torch.Tensor): (batch_size, output_size)
            prediction (list): prediction strings
        """
        logits: torch.Tensor = self.model(x, x_len)
        predictions = [
            self.category.decode(index)
            for index in logits.argmax(dim=-1).detach().cpu().tolist()
        ]
        return logits, predictions

    def forward(
        self,
        _mode: str,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        class_id: torch.LongTensor,
        label: List[str],
        unique_name: List[str],
        _dump_dir: str = None,
    ):
        logits, prediction = self.predict(x, x_len)
        loss = F.cross_entropy(logits, class_id)

        cacheable = dict(
            loss=loss.detach().cpu(),
            prediction=prediction,
            label=[self.category.decode(idx) for idx in class_id],
            unique_name=unique_name,
        )

        return loss, cacheable

    def reduction(self, _mode: str, cached_results: List[dict], _dump_dir: str = None):
        results = self.parse_cached_results(cached_results)
        predictions = results["prediction"]
        labels = results["label"]
        losses = results["loss"]

        acc = accuracy(predictions, labels)
        loss = (sum(losses) / len(losses)).item()

        return dict(
            loss=loss,
            accuracy=acc,
        )


class UtteranceMultiClassClassificationTask(Task):
    def __init__(self, model: UtteranceClassifierExample, categories: CategoryEncoders):
        super().__init__()
        self.model = model
        self.categories = categories
        assert self.model.output_size == len(categories)

    def predict(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            logit (torch.Tensor): List[(batch_size, sub_output_size)]
            prediction (np.array): (batch_size, num_category)
        """
        logits: torch.Tensor = self.model(x, x_len)

        logit_start = 0
        sub_logits, sub_predictions = [], []
        for category in self.categories:
            logit_end = logit_start + len(category)
            sub_logit = logits[:, logit_start:logit_end]
            sub_logits.append(sub_logit)
            sub_predictions.append(
                [
                    category.decode(index)
                    for index in sub_logit.argmax(dim=-1).detach().cpu().tolist()
                ]
            )
            logit_start = logit_end
        prediction = np.array(sub_predictions, dtype="object").T

        return sub_logits, prediction

    def forward(
        self,
        _mode: str,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        class_ids: torch.LongTensor,
        labels: np.ndarray,
        unique_name: List[str],
        _dump_dir: str = None,
    ):
        """
        Args:
            x: torch.Tensor, (batch_size, timestamps, input_size)
            x_len: torch.LongTensor, (batch_size)
            class_ids: torch.LongTensor, (batch_size, num_category)
            labels: np.ndarray, (batch_size, num_category)

        Return:
            loss: torch.Tensor
            prediction: np.ndarray
            label: np.ndarray
        """
        logit, prediction = self.predict(x, x_len)
        loss = sum(
            [
                F.cross_entropy(sub_logit, class_id)
                for sub_logit, class_id in zip(logit, class_ids.T)
            ]
        )

        cacheable = dict(
            loss=loss.detach().cpu(),
            prediction=prediction.tolist(),
            label=labels.tolist(),
            unique_name=unique_name,
        )

        return loss, cacheable

    def reduction(self, _mode: str, cached_results: List[dict], _dump_dir: str = None):
        results = self.parse_cached_results(cached_results)
        losses = results["loss"]
        predictions = results["prediction"]
        labels = results["label"]

        acc = accuracy(predictions, labels)
        loss = (sum(losses) / len(losses)).item()

        return dict(
            loss=loss,
            accuracy=acc,
        )
