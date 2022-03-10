from __future__ import annotations
import logging
from typing import List

import torch
import torch.nn.functional as F

from s3prl.metric import accuracy
from s3prl import Module, Output, Logs

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


class UtteranceClassification(Task):
    """
    Attributes:
        input_size (int): defined by model.input_size
        output_size (int): defined by len(categories)
    """

    def __init__(self, model: UtteranceClassifierExample, categories: List[str]):
        """
        model.output_size should match len(categories)

        Args:
            model (UtteranceClassifier)
            categories (List[str]):
                each object in the list is the final prediction content in str.
                use categories.index(content) to encode as numeric label,
                and decode with categories[index].
        """

        super().__init__()
        self.model = model
        self.categories = categories
        assert self.model.output_size == len(categories)

    @property
    def input_size(self):
        return self.model.input_size

    @property
    def output_size(self):
        return len(self.categories)

    def forward(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            logits (torch.Tensor): (batch_size, timestamps, output_size)
            prediction (list): prediction strings
        """
        logits: torch.Tensor = self.model(x, x_len).slice(1)
        predictions = [
            self.categories[index]
            for index in logits.argmax(dim=-1).detach().cpu().tolist()
        ]
        return Output(logit=logits, prediction=predictions)

    def _general_forward(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        label: List[str],
        name: List[str],
    ):
        logits, prediction = self(x, x_len).slice(2)
        y = torch.LongTensor([self.categories.index(l) for l in label]).to(x.device)
        loss = F.cross_entropy(logits, y)

        logs = Logs()
        logs.add_hidden_state("logits", logits)

        return Output(
            loss=loss,
            prediction=prediction,
            label=label,
            name=name,
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

    def train_step(self, x: torch.Tensor, x_len: torch.LongTensor, label: List[str], name: List[str]):
        """
        Each forward step in the training loop

        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            label (List[str])
            name (List[str])

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
        return self._general_forward(x, x_len, label, name)

    def train_reduction(self, batch_results: list, on_epoch_end: bool = None):
        """
        After several forward steps, outputs should be collected untouched (but detaching the Tensors)
        into a list and passed as batch_results. This function examine the collected items and compute
        metrics across these batches. This function might be called in the middle of an epoch for quick
        logging, or after exactly an epoch to know the epoch level performance.

        Args:
            batch_results (List[cacheable version of the output of self.trainint_step])
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

    def valid_step(self, x: torch.Tensor, x_len: torch.LongTensor, label: List[str], name: List[str]):
        return self._general_forward(x, x_len, label, name)

    def test_step(self, x: torch.Tensor, x_len: torch.LongTensor, label: List[str], name: List[str]):
        return self._general_forward(x, x_len, label, name)

    def valid_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)
