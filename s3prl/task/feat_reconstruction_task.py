from __future__ import annotations

import logging
from typing import List

import torch

from s3prl import Logs, Output
from s3prl.nn.predictor_mockingjay import PredictorMockingjay as PredictorExample
from s3prl.nn.transformer_mockingjay import TransformerMockingjay as UpstreamExample

from . import Task

logger = logging.getLogger(__name__)


class FeatReconstructionTask(Task):
    """
    Attributes:
        upstream (torch.nn.Module): The upstream encoder (transformers, rnn, etc) that outputs `hidden_states`
        predictor (torch.nn.Module): The pre-training predictor that takes `hidden_states` as input and maps to the task target
        loss (torch.nn Loss Functions): The reconstruction loss (torch.nn.L1Loss, torch.nn.MSELoss, etc)
    """

    def __init__(
        self,
        upstream: UpstreamExample,
        predictor: PredictorExample,
        loss: torch.nn.L1Loss,
        loss_config: dict = {},
        **kwargs,
    ):
        """
        The input feature does not necessary have to be the same as the target feature.

        Args:
            upstream (Encoder)
            predictor (Projection NN)
            loss (reconstruction loss)
                feat_A -> upstream -> predictor -> feat_B
                loss(feat_A, feat_B)
        """

        super().__init__()
        self.upstream = upstream
        self.predictor = predictor
        self.loss = loss(**loss_config)

    def predict(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        label_mask: torch.BoolTensor = None,
        position_encoding: torch.Tensor = None,
        attention_mask: torch.LongTensor = None,
    ):
        """
        Args:
            x (torch.Tensor): source_feat - (batch_size, timestamps, input_size)
            label (torch.Tensor): target_feat - (batch_size, timestamps, output_size)
            label_mask (torch.BoolTensor): (batch_size, timestamps, output_size)
            position_encoding (torch.Tensor): (batch_size, timestamps, input_size)
            attention_mask (torch.LongTensor): (batch_size, timestamps)

        Return:
            hidden_states (torch.Tensor): (batch_size, timestamps, hidden_size)
            loss (torch.Tensor): scalar.
            prediction (torch.Tensor): (batch_size, timestamps, output_size)
        """
        if position_encoding is None and attention_mask is None:
            upstream_output: torch.Tensor = self.upstream(x)
        else:
            upstream_output: torch.Tensor = self.upstream(
                x, position_encoding, attention_mask
            )
        prediction: torch.Tensor = self.predictor(upstream_output).prediction

        if label_mask is None:
            reconstruction_loss = self.loss(prediction, label)
        else:
            assert label_mask.sum() > 0, "Without any masking, loss might go NaN."
            reconstruction_loss = self.loss(
                prediction.masked_select(label_mask), label.masked_select(label_mask)
            )

        return Output(
            loss=reconstruction_loss,
            hidden_states=upstream_output.hidden_states,
            prediction=prediction,
        )

    def _general_forward(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        label_mask: torch.BoolTensor = None,
        position_encoding: torch.Tensor = None,
        attention_mask: torch.LongTensor = None,
        unique_name: List[str] = None,
    ):

        loss, hidden_states, prediction = self.predict(
            x, label, label_mask, position_encoding, attention_mask
        ).slice(3)

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
        x: torch.Tensor,
        label: torch.Tensor,
        label_mask: torch.BoolTensor = None,
        position_encoding: torch.Tensor = None,
        attention_mask: torch.LongTensor = None,
        unique_name: List[str] = None,
        **kwargs,
    ):
        """
        Each forward step in the training loop

        Args:
            source_feat (torch.Tensor): (batch_size, timestamps, input_size)
            target_feat (torch.Tensor): (batch_size, timestamps, output_size)
            label_mask (torch.BoolTensor): (batch_size, timestamps, output_size)
            pos_enc (torch.Tensor): (batch_size, timestamps, input_size)
            attn_mask (torch.LongTensor): (batch_size, timestamps)

        Return:
            hidden_states (torch.Tensor): (batch_size, timestamps, hidden_size)
            loss (torch.Tensor): scalar.
            prediction (torch.Tensor): (batch_size, timestamps, output_size)
        """
        return self._general_forward(
            x, label, label_mask, position_encoding, attention_mask, unique_name
        )

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
        x: torch.Tensor,
        label: torch.Tensor,
        label_mask: torch.BoolTensor = None,
        position_encoding: torch.Tensor = None,
        attention_mask: torch.LongTensor = None,
        unique_name: List[str] = None,
        **kwargs,
    ):
        return self._general_forward(
            x, label, label_mask, position_encoding, attention_mask, unique_name
        )

    def test_step(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        label_mask: torch.BoolTensor = None,
        position_encoding: torch.Tensor = None,
        attention_mask: torch.LongTensor = None,
        unique_name: List[str] = None,
        **kwargs,
    ):
        return self._general_forward(
            x, label, label_mask, position_encoding, attention_mask, unique_name
        )

    def valid_reduction(self, batch_results: list, on_epoch_end: bool = True):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_reduction(self, batch_results: list, on_epoch_end: bool = True):
        return self._general_reduction(batch_results, on_epoch_end)
