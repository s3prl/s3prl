from __future__ import annotations

import logging
from typing import List

import torch

from s3prl import Logs, Output
from s3prl.nn.identity import Identity as HeadExample
from s3prl.nn.rnn_apc import APC as BodyExample

from . import Task

logger = logging.getLogger(__name__)


class AutoregressiveReconstructionTask(Task):
    """
    Attributes:
        body (torch.nn.Module): The upstream encoder (transformers, rnn, etc) that outputs `hidden_states`
        head (torch.nn.Module): The pre-training head that takes `hidden_states` as input and maps to the task target
        loss (torch.nn Loss Functions): The reconstruction loss (torch.nn.L1Loss, torch.nn.MSELoss, etc)
    """

    def __init__(
        self, body: BodyExample, head: HeadExample, loss: torch.nn.L1Loss, **kwargs
    ):
        """
        The input feature does not necessary have to be the same as the target feature.

        Args:
            body (Encoder)
            head (Predictor)
            loss (reconstruction loss)
                feat_A -> body -> head -> feat_B
                loss(feat_A, feat_B)
        """

        super().__init__()
        self.body = body
        self.head = head
        self.loss = loss

    def forward(
        self,
        source_feat: torch.Tensor,
        target_feat: torch.Tensor,
        feat_len: int,
    ):
        """
        Args:
            source_feat (torch.Tensor): (batch_size, timestamps, input_size)
            target_feat (torch.Tensor): (batch_size, timestamps, output_size)
            feat_len (int): length of the original feature sequence minus `n_future`

        Return:
            hidden_states (torch.Tensor): (batch_size, timestamps, hidden_size)
            loss (torch.Tensor): scalar.
            prediction (torch.Tensor): (batch_size, timestamps, output_size)
        """
        body_output: torch.Tensor = self.body(source_feat, feat_len.tolist())
        prediction: torch.Tensor = self.head(body_output).prediction

        reconstruction_loss = self.loss(prediction, target_feat)

        return Output(
            loss=reconstruction_loss,
            hidden_states=body_output.hidden_states,
            prediction=prediction,
        )

    def _general_forward(
        self,
        x: torch.Tensor,  # source_feat
        label: torch.Tensor,  # target_feat
        x_len: int,  # feat_len
        unique_name: List[str],
    ):

        loss, hidden_states, prediction = self(x, label, x_len).slice(3)

        logs = Logs()
        logs.add_hidden_state("hidden_states", hidden_states)
        logs.add_hidden_state("prediction", prediction)

        return Output(
            loss=loss,
            prediction=prediction,
            label=label,
            unique_name=unique_name,
            logs=logs,
        )

    def _general_reduction(self, batch_results: list, on_epoch_end: bool = None):
        losses = []
        for batch_result in batch_results:
            losses.append(batch_result.loss)

        loss = (sum(losses) / len(losses)).item()

        logs = Logs()
        logs.add_scalar("loss", loss)

        return Output(
            logs=logs,
        )

    def train_step(
        self,
        x: torch.Tensor,  # source_feat
        label: torch.Tensor,  # target_feat
        x_len: int,  # feat_len
        unique_name: List[str],
        **kwargs,
    ):
        """
        Each forward step in the training loop

        Args:
            source_feat (torch.Tensor): (batch_size, timestamps, input_size)
            target_feat (torch.Tensor): (batch_size, timestamps, output_size)
            feat_len (int): length of the original feature sequence minus `n_future`

        Return:
            hidden_states (torch.Tensor): (batch_size, timestamps, hidden_size)
            loss (torch.Tensor): scalar.
            prediction (torch.Tensor): (batch_size, timestamps, output_size)
        """
        return self._general_forward(x, label, x_len, unique_name)

    def train_reduction(self, batch_results: list, on_epoch_end: bool = False):
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
        x: torch.Tensor,  # source_feat
        label: torch.Tensor,  # target_feat
        x_len: int,  # feat_len
        unique_name: List[str],
        **kwargs,
    ):
        return self._general_forward(x, label, x_len, unique_name)

    def test_step(
        self,
        x: torch.Tensor,  # source_feat
        label: torch.Tensor,  # target_feat
        x_len: int,  # feat_len
        unique_name: List[str],
        **kwargs,
    ):
        return self._general_forward(x, label, x_len, unique_name)

    def valid_reduction(self, batch_results: list, on_epoch_end: bool = True):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_reduction(self, batch_results: list, on_epoch_end: bool = True):
        return self._general_reduction(batch_results, on_epoch_end)
