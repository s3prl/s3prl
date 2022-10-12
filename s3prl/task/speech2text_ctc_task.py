"""
Speech2Text with CTC loss

Authors
  * Heng-Jui Chang 2022
"""

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from s3prl.dataio.encoder.tokenizer import Tokenizer
from s3prl.metric import cer, per, wer
from s3prl.metric.slot_filling import (
    slot_edit_f1_full,
    slot_edit_f1_part,
    slot_type_f1,
    slot_value_cer,
    slot_value_wer,
)
from s3prl.nn import BeamDecoder

from . import Task

logger = logging.getLogger(__name__)

__all__ = [
    "Speech2TextCTCExample",
    "Speech2TextCTCTask",
]


class Speech2TextCTCExample(nn.Module):
    """An example speech-to-text task with CTC objective

    Args:
        input_size (int, optional): Input size. Defaults to 3.
        output_size (int, optional): Output size. Defaults to 4.
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
            y (torch.Tensor): (batch_size, output_size)
            y_len (torch.LongTensor): (batch_size)
        """
        assert x.size(-1) == self.input_size
        output = torch.randn(x.size(0), x.size(1), self.output_size)
        assert output, x_len


class Speech2TextCTCTask(Task):
    """Speech-to-text task with CTC objective

    Args:
        model (Speech2TextCTCExample)
        tokenizer (Tokenizer): Text tokenizer.
        decoder (Union[BeamDecoder, dict], optional):
            Beam decoder or decoder's config. Defaults to None.
        log_metrics (List[str], optional):
            Metrics to be logged. Defaults to ["cer", "wer"].
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        decoder: Union[BeamDecoder, dict] = None,
        log_metrics: List[str] = ["cer", "wer"],
    ) -> None:
        super().__init__()
        self.model = model

        assert isinstance(tokenizer, Tokenizer)
        self.tokenizer = tokenizer
        self.log_metrics = log_metrics

        if BeamDecoder is None:
            decoder = None
        if isinstance(decoder, dict):
            decoder = BeamDecoder(**decoder)
            logger.info("Using flashlight decoder.")
        self.decoder = decoder

        self.criterion = nn.CTCLoss(
            blank=self.tokenizer.pad_idx,
            zero_infinity=True,
        )

    def predict(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            logits (torch.Tensor): (batch_size, timestamps, output_size)
            prediction (list): prediction strings
            valid_length (torch.LongTensor): (batch_size, )
        """

        logits, x_len = self.model(x, x_len)
        predicted_tokens = torch.argmax(logits, dim=2).detach().cpu()
        filtered_tokens = [
            [
                token
                for token in pred_token.unique_consecutive().tolist()
                if token != self.tokenizer.pad_idx and token != self.tokenizer.eos_idx
            ]
            for pred_token in predicted_tokens
        ]
        predictions = [
            self.tokenizer.decode(token_list) for token_list in filtered_tokens
        ]
        return logits, predictions, x_len

    def forward(
        self,
        _mode: str,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        labels: np.ndarray,
        class_ids: torch.LongTensor,
        unique_name: np.ndarray,
        beam_decode: bool = False,
        _dump_dir: str = None,
    ):
        """
        Each forward step in the training loop

        Args:
            mode (str): train / valid / test
            x (torch.Tensor):
                Input waveform or acoustic features.
                (batch_size, timestamps, input_size)
            x_len (torch.LongTensor):
                Lengths of inputs.
                (batch_size, )
            labels (np.ndarray):
                Ground truth transcriptions (str).
                (batch_size, )
            class_ids (torch.LongTensor):
                Tokenized ground truth transcriptions.
            unique_name (np.ndarray):
                Unique names for each sample.

        """
        logits, prediction, x_len = self.predict(x, x_len)
        log_probs = F.log_softmax(logits, dim=2)

        y = class_ids
        y_len = torch.tensor(
            [(ids != self.tokenizer.pad_idx).long().sum() for ids in class_ids],
            dtype=torch.long,
            device=logits.device,
        )

        loss = self.criterion(log_probs.transpose(0, 1), y, x_len, y_len)

        hyps = None
        if beam_decode and self.decoder is not None:
            hyps = self.decoder.decode(log_probs.detach())

        cacheable = dict(
            loss=loss.detach().cpu().item(),
            prediction=prediction,
            label=labels.tolist(),
            unique_name=unique_name.tolist(),
            hypotheses=hyps,
        )

        return loss, cacheable

    def reduction(self, _mode: str, cached_results: List[dict], _dump_dir: str = None):
        results = self.parse_cached_results(cached_results)

        losses = results["loss"]
        predictions = results["prediction"]
        labels = results["label"]
        unique_names = results["unique_name"]

        if _dump_dir is not None:
            with (Path(_dump_dir) / "ref").open("w") as f:
                f.writelines(
                    [f"{uid} {p}\n" for p, uid in zip(predictions, unique_names)]
                )

            with (Path(_dump_dir) / "hyp").open("w") as f:
                f.writelines([f"{uid} {p}\n" for p, uid in zip(labels, unique_names)])

        beam_hyps = None
        if results["hypotheses"][0] is not None:
            beam_hyps = [" ".join(hyp[0].words) for hyp in results["hypotheses"]]

        logs = {}
        logs["loss"] = float(np.mean(losses))

        if "wer" in self.log_metrics:
            logs["wer"] = wer(predictions, labels)
        if "cer" in self.log_metrics:
            logs["cer"] = cer(predictions, labels)
        if "per" in self.log_metrics:
            logs["per"] = per(predictions, labels)
        if "slot_type_f1" in self.log_metrics:
            logs["slot_type_f1"] = slot_type_f1(predictions, labels)
        if "slot_value_cer" in self.log_metrics:
            logs["slot_value_cer"] = slot_value_cer(predictions, labels)
        if "slot_value_wer" in self.log_metrics:
            logs["slot_value_wer"] = slot_value_wer(predictions, labels)
        if "slot_edit_f1_full" in self.log_metrics:
            logs["slot_edit_f1_full"] = slot_edit_f1_full(predictions, labels)
        if "slot_edit_f1_part" in self.log_metrics:
            logs["slot_edit_f1_part"] = slot_edit_f1_part(predictions, labels)

        if beam_hyps is not None:
            logs["wer_beam"] = wer(beam_hyps, labels)
            logs["char_beam"] = cer(beam_hyps, labels)

        return logs
