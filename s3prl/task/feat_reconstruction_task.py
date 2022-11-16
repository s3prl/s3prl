"""
Feature Reconstruction Task

Authors
  * Andy T. Liu 2022
"""
from __future__ import annotations

import logging
from typing import List

import torch

from s3prl.nn.predictor_mockingjay import PredictorMockingjay as PredictorExample
from s3prl.nn.transformer_mockingjay import TransformerMockingjay as UpstreamExample

from . import Task

logger = logging.getLogger(__name__)

__all__ = [
    "FeatReconstructionTask",
]


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
            upstream (UpstreamExample): Encoder
            predictor (PredictorExample): Projection NN
            loss (torch.nn.L1Loss): Reconstruction loss
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
            loss (torch.Tensor): scalar.
            hidden_states (torch.Tensor): (batch_size, timestamps, hidden_size)
            prediction (torch.Tensor): (batch_size, timestamps, output_size)
        """
        if position_encoding is None and attention_mask is None:
            hidden_states: torch.Tensor = self.upstream(x)
        else:
            hidden_states: torch.Tensor = self.upstream(
                x, position_encoding, attention_mask
            )
        prediction: torch.Tensor = self.predictor(hidden_states)

        if label_mask is None:
            reconstruction_loss = self.loss(prediction, label)
        else:
            assert label_mask.sum() > 0, "Without any masking, loss might go NaN."
            reconstruction_loss = self.loss(
                prediction.masked_select(label_mask), label.masked_select(label_mask)
            )

        return reconstruction_loss, hidden_states, prediction

    def forward(
        self,
        _mode: str,
        x: torch.Tensor,
        label: torch.Tensor,
        label_mask: torch.BoolTensor = None,
        position_encoding: torch.Tensor = None,
        attention_mask: torch.LongTensor = None,
        unique_name: List[str] = None,
        _dump_dir: str = None,
    ):
        loss, hidden_states, prediction = self.predict(
            x, label, label_mask, position_encoding, attention_mask
        )

        cacheable = dict(
            loss=loss.detach().cpu(),
            prediction=prediction,
            label=label,
            hidden_states=hidden_states,
            unique_name=unique_name,
        )

        return loss, cacheable

    def reduction(self, _mode: str, cached_results: List[dict], _dump_dir: str = None):
        results = self.parse_cached_results(cached_results)
        losses = results["loss"]
        predictions = results["prediction"]

        loss = (sum(losses) / len(losses)).item()

        return dict(
            loss=loss,
            prediction=predictions[0],
        )
