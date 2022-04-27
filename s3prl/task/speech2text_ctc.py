import logging
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from s3prl import Logs, Module, Output
from s3prl.encoder.tokenizer import Tokenizer
from s3prl.metric import cer, wer

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


class Speech2TextCTC(Task):
    def __init__(self, model: Speech2TextCTCExample, tokenizer: Tokenizer) -> None:
        """Speech-to-text task with CTC objective

        Args:
            model (Speech2TextCTCExample)
            tokenizer (Tokenizer): Text tokenizer.
        """

        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        assert self.model.output_size == self.tokenizer.vocab_size

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

    def forward(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            logits (torch.Tensor): (batch_size, timestamps, output_size)
            prediction (list): prediction strings
        """

        logits, x_len = self.model(x, x_len).slice(2)
        logits_argmax = torch.argmax(logits, dim=2).detach().cpu()
        predictions = [
            self.tokenizer.decode(
                logits_argmax[b, : x_len[b]].tolist(), ignore_repeat=True
            )
            for b in range(len(logits_argmax))
        ]
        return Output(logit=logits, prediction=predictions, output_size=x_len)

    def _general_forward(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        label: List[int],
        name: List[str],
    ):
        logits, prediction, x_len = self(x, x_len).slice(3)
        log_probs = F.log_softmax(logits, dim=2)
        label_len = torch.tensor(
            [len(l) for l in label], dtype=torch.long, device=logits.device
        )
        label = pad_sequence(label, batch_first=True).to(logits.device)

        loss = self.criterion(log_probs.transpose(0, 1), label, x_len, label_len)

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
        labels = [self.tokenizer.decode(label) for label in labels]

        word_error_rate = wer(predictions, labels)
        char_error_rate = cer(predictions, labels)
        loss = (sum(losses) / len(losses)).item()

        logs = Logs()
        logs.add_scalar("loss", loss)
        logs.add_scalar("wer", word_error_rate)
        logs.add_scalar("cer", char_error_rate)

        return Output(
            logs=logs,
        )

    def train_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        label: List[int],
        name: List[str],
    ):
        """
        Each forward step in the training loop

        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )
            label (List[int])
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

    def valid_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        label: List[int],
        name: List[str],
    ):
        return self._general_forward(x, x_len, label, name)

    def test_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        label: List[int],
        name: List[str],
    ):
        return self._general_forward(x, x_len, label, name)

    def train_reduction(self, batch_results: list, on_epoch_end: bool = None):
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

    def valid_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)
