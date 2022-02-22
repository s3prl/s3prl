from __future__ import annotations
import logging
from typing import List
from pathlib import Path
from functools import partial
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from s3prl import init, Module, Output
from s3prl.metric import accuracy
from s3prl.util import Log, LogDataType

from . import Task

logger = logging.getLogger(__name__)


class UtteranceClassifier(Module):
    """
    Attributes:
        input_size: int
        output_size: int
    """
    @init.method
    def __init__(self, input_size=3, output_size=4):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch_size, timestemps, input_size)

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
    """

    @init.method
    def __init__(self, model: UtteranceClassifier, categories: List[str]):
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

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)

        Return:
            logits (torch.Tensor): (batch_size, timestamps, output_size)
            prediction (list): prediction strings
        """
        logits: torch.Tensor = self.model(x)
        return Output(logits=logits)

    def inference(self, x: torch.Tensor):
        """
        Decode to human readable: solve the task
        """
        y = self(x).output
        # decoding
        decode = self.decoded(y)
        pass

    def _general_forward(self, x: torch.Tensor, label: list, batch_idx: int):
        logits, prediction = self(x)
        y = torch.LongTensor([self.categories.index(l) for l in label]).to(x.device)
        loss = F.cross_entropy(logits, y)

        def callback_saver(dir: str, logits: list):
            torch.save(dict(logits=logits), Path(dir) / f"{batch_idx}.logits")

        return Output(
            loss=loss,
            saver=partial(callback_saver, logits=logits.detach().cpu()),
            prediction=prediction,
            label=label,
        )

    def _general_reduction(self, batch_results: list, on_epoch_end: bool = None):
        predictions, labels, losses = [], [], []
        for batch_result in batch_results:
            predictions += batch_result.prediction
            labels += batch_result.label
            losses.append(batch_result.loss)

        acc = accuracy(predictions, labels)
        loss = (sum(losses) / len(losses)).item()

        return Output(
            logs=[
                Log("loss", loss, LogDataType.SCALAR),
                Log("accuracy", acc, LogDataType.SCALAR),
            ],
        )

    def training_step(self, x: torch.Tensor, label: list, batch_idx: int):
        """
        Each forward step in the training loop

        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            label (List[str])
            batch_idx (int):
                the current batch id among the entire dataloader
                this can be useful for logging

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
        return self._general_forward(x, label)

    def training_reduction(self, batch_results: list, on_epoch_end: bool = None):
        """
        After several forward steps, outputs should be collected untouched (but detaching the Tensors)
        into a list and passed as batch_results. This function examine the collected items and compute
        metrics across these batches. This function might be called in the middle of an epoch for quick
        logging, or after exactly an epoch to know the epoch level performance.

        Args:
            batch_results (List[detached version of self.trainint_step])
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

    def validation_step(self, x: torch.Tensor, label: list, batch_idx: int):
        return self._general_forward(x, label)

    def test_step(self, x: torch.Tensor, label: list, batch_idx: int):
        return self._general_forward(x, label)

    def validation_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)
