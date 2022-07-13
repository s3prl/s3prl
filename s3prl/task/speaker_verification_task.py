from __future__ import annotations

import logging
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from s3prl import Logs, Module, Output, Container
from s3prl.metric import accuracy, compute_eer, compute_minDCF
from s3prl.nn import amsoftmax, softmax

from . import Task

logger = logging.getLogger(__name__)


class SpeakerClassifier(Module):
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


class SpeakerVerification(Task):
    """
    model.output_size should match len(categories)

    Args:
        model (SpeakerClassifier):
            actual model or a callable config for the model
        categories (dict[str]):
            each key in the Dictionary is the final prediction content in str.
            use categories[key] to encode as numeric label
        trials (List[Tuple[int, str, str]]):
            each tuple in the list consists of (label, enroll_utt, test_utt)
        loss_type (str): softmax or amsoftmax
        loss_cfg (dict): **kwds for loss_type class
    """

    def __init__(
        self,
        model: SpeakerClassifier,
        categories: Dict[str, int],
        trials: List[Tuple[int, str, str]],
        loss_type: str = "amsoftmax",
        loss_cfg: dict = None,
        **unused,
    ):
        super().__init__()
        self.model = Container(model)() if isinstance(model, dict) else model
        self.categories = categories
        self.trials = trials

        if loss_type == "amsoftmax":
            loss_cls = amsoftmax
        elif loss_type == "softmax":
            loss_cls = softmax
        else:
            raise ValueError(f"Unsupported loss_type {loss_type}")

        self.loss = loss_cls(
            input_size=self.model.output_size,
            output_size=len(self.categories),
            **loss_cfg,
        )
        assert self.loss.output_size == len(categories)

    @property
    def input_size(self):
        return self.model.input_size

    @property
    def output_size(self):
        return len(self.categories)

    def predict(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            hidden_states (torch.Tensor): (batch_size, output_size)
            hidden_states_len (list): (batch_size, )
        """
        spk_embeddings = self.model(x, x_len).slice(1)

        return Output(hidden_states=spk_embeddings)

    def _general_forward(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        label: torch.LongTensor,
        unique_name: List[str],
        **unused,
    ):
        spk_embeddings = self.predict(x, x_len).slice(1)
        loss, logits = self.loss(spk_embeddings, label).slice(2)

        prediction = [index for index in logits.argmax(dim=-1).detach().cpu().tolist()]

        logs = Logs()
        logs.add_hidden_state("logits", logits)

        return Output(
            loss=loss,
            prediction=prediction,
            label=label,
            unique_name=unique_name,
            logs=logs,
        )

    def _general_reduction(
        self, batch_results: list, on_epoch_end: bool = None, **unused
    ):
        predictions, labels, losses = [], [], []
        for batch_result in batch_results:
            predictions += batch_result.prediction
            labels += list(batch_result.label.cpu().numpy())
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
        label: torch.LongTensor,
        unique_name: List[str],
        **unused,
    ):
        """
        Each forward step in the training loop

        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len: torch.LongTensor
            label: torch.LongTensor
            unique_name (List[str])

        Return:
            loss (torch.Tensor):
                undetached loss. When using this module. Please sanitize this loss before
                collecting the returning Namespace into a list for future aggregation/reduction
            prediction (List[int])
            label: torch.LongTensor
            unique_name (List[str])
            logits:
                this is not required, just to let the end-user get as many info as possible
                so that people can do more things with the internal states
        """
        return self._general_forward(x, x_len, label, unique_name)

    def train_reduction(self, batch_results: list, on_epoch_end: bool = None, **unused):
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
        label: torch.LongTensor,
        unique_name: List[str],
        **unused,
    ):
        return self._general_forward(x, x_len, label, unique_name)

    def valid_reduction(
        self,
        batch_results: list,
        on_epoch_end: bool = None,
        **unused,
    ):
        return self._general_reduction(batch_results, on_epoch_end)

    def test_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        unique_name: List[str],
        **unused,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len: torch.LongTensor
            unique_name (List[str])

        Return:
            unique_name (List[str])
            output (torch.Tensor):
                speaker embeddings corresponding to unique_name
        """
        spk_embeddings = self.predict(x, x_len).slice(1)
        return Output(unique_name=unique_name, output=spk_embeddings)

    def test_reduction(
        self, batch_results: list(), on_epoch_end: bool = None, **unused
    ):

        embeddings = {}
        for batch_result in batch_results:
            for key, value in zip(batch_result.unique_name, batch_result.output):
                embeddings[key] = value

        trials = self.trials
        scores = []
        labels = []
        for label, enroll, test in tqdm(trials, desc="Test Scoring", total=len(trials)):
            enroll_embd = embeddings[enroll]
            test_embd = embeddings[test]
            score = F.cosine_similarity(enroll_embd, test_embd, dim=0).item()
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
