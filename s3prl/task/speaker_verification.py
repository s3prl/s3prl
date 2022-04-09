from __future__ import annotations
import logging
from typing import List
from tqdm import tqdm

import torch
import torch.nn.functional as F

from s3prl.metric import accuracy, compute_eer, compute_minDCF
from s3prl.nn import amsoftmax, softmax
from s3prl import Module, Output, Logs

from . import Task

logger = logging.getLogger(__name__)


class SpeakerVerification(Module):
    """
    Attributes:
        input_size (int): defined by model.input_size
        output_size (int): defined by len(categories)
    """

    def __init__(
        self,
        model,
        categories: dict(str),
        trials: list(tuple()),
        loss_type: str = "softmax",
    ):
        """
        model.output_size should match len(categories)

        Args:
            model (SpeakerClassifier)
            categories (dict[str]):
                each key in the Dictionary is the final prediction content in str.
                use categories[key] to encode as numeric label
            trials:
                each tuple in the list consists of (enroll_path, test_path, label)
        """

        super().__init__()
        self.model = model
        self.categories = categories
        self.trials = trials

        if loss_type == "amsoftmax":
            self.loss = amsoftmax(
                input_size=self.model.output_size, output_size=len(self.categories)
            )

        elif loss_type == "softmax":
            self.loss = softmax(
                input_size=self.model.output_size, output_size=len(self.categories)
            )

        else:
            raise ValueError("{} loss type is not defined".format(loss_type))

        assert self.loss.output_size == len(categories)

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
            hidden_states (torch.Tensor): (batch_size, output_size)
            hidden_states_len (list): (batch_size, )
        """
        # TODO: consider x_len into pooling
        spk_embeddings = self.model(x, x_len).slice(1)

        return Output(hidden_states=spk_embeddings)

    def _general_forward(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        label: List[int],
        name: List[str],
    ):
        spk_embeddings = self(x, x_len).slice(1)
        y = torch.LongTensor(label).to(x.device)
        loss, logits = self.loss(spk_embeddings, y).slice(2)

        prediction = [index for index in logits.argmax(dim=-1).detach().cpu().tolist()]

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
            x_len: torch.LongTensor
            label (List[str])
            name (List[str])

        Return:
            loss (torch.Tensor):
                undetached loss. When using this module. Please sanitize this loss before
                collecting the returning Namespace into a list for future aggregation/reduction
            prediction (List[int])
            label (List[int])
            name (List[str])
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
        label: List[str],
        name: List[str],
    ):
        return self._general_forward(x, x_len, label, name)

    def valid_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        name: List[str],
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len: torch.LongTensor
            label (List[str])
            name (List[str])

        Return:
            name (List[str])
            output (torch.Tensor):
                speaker embeddings corresponding to name
        """
        spk_embeddings = self(x, x_len).slice(1)
        return Output(name=name, output=spk_embeddings)

    def test_reduction(self, batch_results: list(), on_epoch_end: bool = None):

        embeddings = {}
        for batch_result in batch_results:
            for key, value in zip(batch_result.name, batch_result.output):
                embeddings[key] = value

        trials = self.trials
        scores = []
        labels = []
        for label, enroll, test in tqdm(trials, desc="Test Scoring", total=len(trials)):
            enroll_embd = embeddings[enroll]
            test_embd   = embeddings[test]
            score       = F.cosine_similarity(enroll_embd, test_embd, dim=0).item()
            scores.append(score)
            labels.append(label)

        EER, EERthreshold = compute_eer(labels, scores)

        minDCF, minDCFthreshold = compute_minDCF(labels, scores, p_target=0.01)

        logs = Logs()
        logs.add_scalar("EER", EER)
        logs.add_scalar("EERthreshold", EERthreshold.item())
        logs.add_scalar("minDCF", minDCF)
        logs.add_scalar("minDCF_threshold", minDCFthreshold)

        return Output(logs=logs)
