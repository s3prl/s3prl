from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from s3prl import Logs, Module, Output
from s3prl.metric import accuracy

from . import Task

logger = logging.getLogger(__name__)


class UtteranceClassifierExample(Module):
    """
    Attributes:
        input_size: int
        output_size: int
    """

    def __init__(self, input_size=3, output_size=4):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

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
        assert Output(output=output)


class UtteranceClassificationTask(Task):
    """
    Attributes:
        input_size (int): defined by model.input_size
        output_size (int): defined by len(categories)
    """

    def __init__(self, model: UtteranceClassifierExample, category, **kwargs):
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
        self._current_best_acc = 0.0

    @property
    def input_size(self):
        return self.model.input_size

    @property
    def output_size(self):
        return len(self.categories)

    def predict(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            logits (torch.Tensor): (batch_size, output_size)
            prediction (list): prediction strings
        """
        logits: torch.Tensor = self.model(x, x_len).slice(1)
        predictions = [
            self.category.decode(index)
            for index in logits.argmax(dim=-1).detach().cpu().tolist()
        ]
        return Output(logit=logits, prediction=predictions)

    def _general_forward(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        class_id: torch.LongTensor,
        unique_name: List[str],
    ):
        logits, prediction = self.predict(x, x_len).slice(2)
        loss = F.cross_entropy(logits, class_id)

        logs = Logs()
        logs.add_hidden_state("logits", logits)

        return Output(
            loss=loss,
            prediction=prediction,
            label=[self.category.decode(idx) for idx in class_id],
            unique_name=unique_name,
            logs=logs,
        )

    def _general_reduction(self, batch_results: list, on_epoch_end: bool = None):
        predictions, labels, losses = [], [], []
        for batch_result in batch_results:
            predictions += batch_result.prediction
            labels += batch_result.label
            losses.append(batch_result.loss)

        acc = accuracy(predictions, labels)
        loss = (sum(losses) / len(losses)).item()

        logs = Logs()
        logs.add_scalar("loss", loss)
        logs.add_scalar("accuracy", acc)

        return Output(
            logs=logs,
        )

    def train_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        class_id: torch.LongTensor,
        unique_name: List[str],
        **kwargs,
    ):
        """
        Each forward step in the training loop

        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            class_id (List[str])
            unique_name (List[str])

        Return:
            loss (torch.Tensor):
                undetached loss. When using this module. Please sanitize this loss before
                collecting the returning Namespace into a list for future aggregation/reduction
            saver (Callable[str, None]):
                end-user can simply call the saver with a customized directory
                and don't need to take care of how to save the data in to correct format
            logits:
                this is not required, just to let the end-user get as many info as possible
                so that people can do more things with the internal states
        """
        return self._general_forward(x, x_len, class_id, unique_name)

    def train_reduction(self, batch_results: list, on_epoch_end: bool = False, **kwds):
        """
        After several forward steps, outputs should be collected untouched (but detaching the Tensors)
        into a list and passed as batch_results. This function examine the collected items and compute
        metrics across these batches. This function might be called in the middle of an epoch for quick
        logging, or after exactly an epoch to know the epoch level performance.

        Args:
            batch_results (List[cacheable version of the output of self.train_step])
            on_epoch_end (bool):
                usually you should keep the same behavior between sub-epoch and epoch level
                this parameter is here in case you need specific postprocessing which must
                only be done right on the end of an epoch

        Return:
            logs (List[Log]):
                a list of content to log onto any logger
                each content should be in the Log class format
        """
        return self._general_reduction(batch_results, on_epoch_end)

    def valid_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        class_id: torch.LongTensor,
        unique_name: List[str],
        **kwargs,
    ):
        return self._general_forward(x, x_len, class_id, unique_name)

    def test_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        class_id: torch.LongTensor,
        unique_name: List[str],
        **kwargs,
    ):
        return self._general_forward(x, x_len, class_id, unique_name)

    def valid_reduction(self, batch_results: list, on_epoch_end: bool = True, **kwds):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_reduction(self, batch_results: list, on_epoch_end: bool = True, **kwds):
        return self._general_reduction(batch_results, on_epoch_end)


class UtteranceMultiClassClassificationTask(Task):
    def __init__(self, model: UtteranceClassifierExample, categories, **kwargs):
        super().__init__()
        self.model = model
        self.categories = categories
        assert self.model.output_size == sum([len(c) for c in categories])
        self._current_best_acc = 0.0

    @property
    def input_size(self):
        return self.model.input_size

    @property
    def output_size(self):
        return self.model.output_size

    def predict(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            logit (torch.Tensor): List[(batch_size, sub_output_size)]
            prediction (np.array): (batch_size, num_category)
        """
        logits: torch.Tensor = self.model(x, x_len).slice(1)
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

        return Output(logit=sub_logits, prediction=prediction)

    def _general_forward(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        class_ids: torch.LongTensor,
        labels: np.ndarray,
        unique_name: List[str],
        **kwds,
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
        logit, prediction = self.predict(x, x_len).slice(2)
        loss = sum(
            [
                F.cross_entropy(sub_logit, class_id)
                for sub_logit, class_id in zip(logit, class_ids.T)
            ]
        )

        logs = Logs()
        logs.add_hidden_state("logit", logit)

        return Output(
            loss=loss,
            prediction=prediction,
            label=labels,
            unique_name=unique_name,
            logs=logs,
        )

    @staticmethod
    def numpy_object_array_all_close(x, y):
        return not (x != y).sum() > 0

    def _general_reduction(
        self, batch_results: list, on_epoch_end: bool = None, **kwds
    ):
        losses, predictions, labels = [], [], []
        for batch_result in batch_results:
            predictions += list(batch_result.prediction)
            labels += list(batch_result.label)
            losses.append(batch_result.loss)

        acc = accuracy(
            predictions, labels, item_same_fn=self.numpy_object_array_all_close
        )
        loss = (sum(losses) / len(losses)).item()

        logs = Logs()
        logs.add_scalar("loss", loss)
        logs.add_scalar("accuracy", acc)

        return Output(
            logs=logs,
        )

    def train_step(self, *args, **kwargs):
        return self._general_forward(*args, **kwargs)

    def train_reduction(self, batch_results: list, on_epoch_end: bool = False, **kwds):
        return self._general_reduction(batch_results, on_epoch_end)

    def valid_step(self, *args, **kwargs):
        return self._general_forward(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self._general_forward(*args, **kwargs)

    def valid_reduction(self, batch_results: list, on_epoch_end: bool = True, **kwds):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_reduction(self, batch_results: list, on_epoch_end: bool = True, **kwds):
        return self._general_reduction(batch_results, on_epoch_end)
