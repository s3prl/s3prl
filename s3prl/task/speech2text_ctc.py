from __future__ import annotations
import logging
from typing import List

import torch
import torch.nn.functional as F

from s3prl.metric import wer, cer, per
from s3prl import Module, Output, Logs

from . import Task

logger = logging.getLogger(__name__)


class Speech2TextCTC(Task):
    def __init__(self) -> None:
        super().__init__()

    def train_step(self):
        raise NotImplementedError

    def valid_step(self):
        raise NotImplementedError

    def test_step(self):
        raise NotImplementedError

    def train_reduction(self, batch_results: list, on_epoch_end: bool = None):
        raise NotImplementedError

    def valid_reduction(self, batch_results: list, on_epoch_end: bool = None):
        raise NotImplementedError

    def test_reduction(self, batch_results: list, on_epoch_end: bool = None):
        raise NotImplementedError
