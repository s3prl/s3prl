"""
Speaker Verification with Softmax-based loss

Authors
  * Po-Han Chi 2021
  * Haibin Wu 2022
"""

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from s3prl.dataio.encoder.category import CategoryEncoder
from s3prl.metric import accuracy, compute_eer, compute_minDCF
from s3prl.nn import amsoftmax, softmax

from . import Task

logger = logging.getLogger(__name__)


__all__ = ["SpeakerClassifier", "SpeakerVerification"]


class SpeakerClassifier(torch.nn.Module):
    """
    Attributes:
        input_size: int
        output_size: int
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
            output (torch.Tensor): (batch_size, output_size)
        """
        assert x.size(-1) == self.input_size
        output = torch.randn(x.size(0), self.output_size)
        assert output


class SpeakerVerification(Task):
    """
    model.output_size should match len(categories)

    Args:
        model (SpeakerClassifier):
            actual model or a callable config for the model
        categories (dict[str]):
            each key in the Dictionary is the final prediction content in str.
            use categories[key] to encode as numeric label
        test_trials (List[Tuple[int, str, str]]):
            each tuple in the list consists of (label, enroll_utt, test_utt)
        loss_type (str): softmax or amsoftmax
        loss_conf (dict): arguments for the loss_type class
    """

    def __init__(
        self,
        model: SpeakerClassifier,
        category: CategoryEncoder,
        test_trials: List[Tuple[int, str, str]] = None,
        loss_type: str = "amsoftmax",
        loss_conf: dict = None,
    ):
        super().__init__()
        self.model = model
        self.category = category
        self.trials = test_trials

        if loss_type == "amsoftmax":
            loss_cls = amsoftmax
        elif loss_type == "softmax":
            loss_cls = softmax
        else:
            raise ValueError(f"Unsupported loss_type {loss_type}")

        self.loss: torch.nn.Module = loss_cls(
            input_size=self.model.output_size,
            output_size=len(self.category),
            **loss_conf,
        )
        assert self.loss.output_size == len(category)

    def get_state(self):
        return {
            "loss_state": self.loss.state_dict(),
        }

    def set_state(self, state: dict):
        self.loss.load_state_dict(state["loss_state"])

    def predict(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, timestamps, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Return:
            torch.Tensor

            (batch_size, output_size)
        """
        spk_embeddings = self.model(x, x_len)
        return spk_embeddings

    def train_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        class_id: torch.LongTensor,
        unique_name: List[str],
        _dump_dir: str = None,
    ):
        spk_embeddings = self.predict(x, x_len)
        loss, logits = self.loss(spk_embeddings, class_id)
        prediction = [index for index in logits.argmax(dim=-1).detach().cpu().tolist()]

        cacheable = dict(
            loss=loss.detach().cpu().item(),
            class_id=class_id.detach().cpu().tolist(),
            prediction=prediction,
            unique_name=unique_name,
        )

        return loss, cacheable

    def train_reduction(self, cached_results: list, _dump_dir: str = None):
        results = self.parse_cached_results(cached_results)
        acc = accuracy(results["prediction"], results["class_id"])
        loss = torch.FloatTensor(results["loss"]).mean().item()

        return dict(
            loss=loss,
            accuracy=acc,
        )

    def test_step(
        self,
        x: torch.Tensor,
        x_len: torch.LongTensor,
        unique_name: List[str],
        _dump_dir: str,
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
        spk_embeddings = self.predict(x, x_len)

        cacheable = dict(
            unique_name=unique_name.tolist(),
            spk_embedding=spk_embeddings.detach().cpu().unbind(dim=0),
        )
        return None, cacheable

    def test_reduction(self, cached_results: List[dict], _dump_dir: str):
        results = self.parse_cached_results(cached_results)
        embeddings = {}
        for name, emb in zip(results["unique_name"], results["spk_embedding"]):
            embeddings[name] = emb

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

        return dict(
            EER=EER,
            EERthreshold=EERthreshold.item(),
            minDCF=minDCF,
            minDCF_threshold=minDCFthreshold,
        )
