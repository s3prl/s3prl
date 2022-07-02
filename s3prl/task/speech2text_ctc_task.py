import logging
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from s3prl import Logs, Module, Output
from s3prl.encoder.tokenizer import Tokenizer
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


class Speech2TextCTCExample(Module):
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
        output = torch.randn(x.size(0), x.size(1), self.output_size)
        assert Output(output=output)


class Speech2TextCTCTask(Task):
    def __init__(
        self,
        model: Speech2TextCTCExample,
        tokenizer: Tokenizer,
        decoder: Union[BeamDecoder, dict] = None,
        log_metrics: List[str] = ["cer", "wer"],
        **kwargs,
    ) -> None:
        """Speech-to-text task with CTC objective

        Args:
            model (Speech2TextCTCExample)
            tokenizer (Tokenizer): Text tokenizer.
            decoder (Union[BeamDecoder, dict], optional):
                Beam decoder or decoder's config. Defaults to None.
            log_metrics (List[str], optional):
                Metrics to be logged. Defaults to ["cer", "wer"].
        """

        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.log_metrics = log_metrics
        assert self.model.output_size == self.tokenizer.vocab_size

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
            logits (torch.Tensor): (batch_size, timestamps, output_size)
            prediction (list): prediction strings
        """

        logits, x_len = self.model(x, x_len).slice(2)
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
        return Output(logit=logits, prediction=predictions, output_size=x_len)

    def _general_forward(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        labels: np.ndarray,
        class_ids: torch.LongTensor,
        unique_name: np.ndarray,
        beam_decode: bool = False,
    ):
        logits, prediction, x_len = self.predict(x, x_len).slice(3)
        log_probs = F.log_softmax(logits, dim=2)

        y = class_ids
        y_len = torch.tensor(
            [(ids != self.tokenizer.pad_idx).long().sum() for ids in class_ids],
            dtype=torch.long,
            device=logits.device,
        )

        loss = self.criterion(log_probs.transpose(0, 1), y, x_len, y_len)

        logs = Logs()
        logs.add_hidden_state("logits", logits)

        hyps = None
        if beam_decode and self.decoder is not None:
            hyps = self.decoder.decode(log_probs.detach())

        return Output(
            loss=loss,
            prediction=prediction,
            labels=labels.tolist(),
            unique_name=unique_name,
            logs=logs,
            hypotheses=hyps,
        )

    def _general_reduction(self, batch_results: list, on_epoch_end: bool = None):
        predictions, labels, losses, beam_hyps = [], [], [], []
        for batch_result in batch_results:
            predictions += batch_result.prediction
            labels += batch_result.labels
            losses.append(batch_result.loss.item())
            if batch_result.hypotheses is not None:
                beam_hyps += [" ".join(hyp[0].words) for hyp in batch_result.hypotheses]

        if self.tokenizer.token_type in {"character-slot", "subword-slot"}:
            labels = [
                self.tokenizer.decode(self.tokenizer.encode(label)) for label in labels
            ]

        logs = Logs()
        logs.add_scalar("loss", np.mean(losses))

        if "wer" in self.log_metrics:
            logs.add_scalar("wer", wer(predictions, labels))
        if "cer" in self.log_metrics:
            logs.add_scalar("cer", cer(predictions, labels))
        if "per" in self.log_metrics:
            logs.add_scalar("per", per(predictions, labels))
        if "slot_type_f1" in self.log_metrics:
            logs.add_scalar("slot_type_f1", slot_type_f1(predictions, labels))
        if "slot_value_cer" in self.log_metrics:
            logs.add_scalar("slot_value_cer", slot_value_cer(predictions, labels))
        if "slot_value_wer" in self.log_metrics:
            logs.add_scalar("slot_value_wer", slot_value_wer(predictions, labels))
        if "slot_edit_f1_full" in self.log_metrics:
            logs.add_scalar("slot_edit_f1_full", slot_edit_f1_full(predictions, labels))
        if "slot_edit_f1_part" in self.log_metrics:
            logs.add_scalar("slot_edit_f1_part", slot_edit_f1_part(predictions, labels))

        if len(beam_hyps) > 0:
            logs.add_scalar("wer_beam", wer(beam_hyps, labels))
            logs.add_scalar("char_beam", cer(beam_hyps, labels))

        return Output(
            logs=logs,
        )

    def train_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        labels: np.ndarray,
        class_ids: torch.LongTensor,
        unique_name: np.ndarray,
    ):
        """
        Each forward step in the training loop

        Args:
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
        return self._general_forward(x, x_len, labels, class_ids, unique_name)

    def valid_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        labels: np.ndarray,
        class_ids: torch.LongTensor,
        unique_name: np.ndarray,
        **kwds,
    ):
        return self._general_forward(x, x_len, labels, class_ids, unique_name)

    def test_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        labels: np.ndarray,
        class_ids: torch.LongTensor,
        unique_name: np.ndarray,
        **kwds,
    ):
        return self._general_forward(
            x,
            x_len,
            labels,
            class_ids,
            unique_name,
            beam_decode=self.decoder is not None,
        )

    def train_reduction(self, batch_results: list, on_epoch_end: bool = None, **kwds):
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

    def valid_reduction(self, batch_results: list, on_epoch_end: bool = None, **kwds):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_reduction(self, batch_results: list, on_epoch_end: bool = None, **kwds):
        return self._general_reduction(batch_results, on_epoch_end)