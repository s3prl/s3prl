# Copyright Hear Benchmark Team
# Copyright Shu-wen Yang (refactor from https://github.com/hearbenchmark/hear-eval-kit)

from typing import List

import torch

from s3prl.dataio.encoder.category import CategoryEncoder
from s3prl.task.base import Task

from ._hear_score import available_scores, validate_score_return_type

__all__ = ["ScenePredictionTask"]


class OneHotToCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.all(torch.sum(y, dim=1) == y.new_ones(y.shape[0]))
        y = y.argmax(dim=1)
        return self.loss(y_hat, y)


class ScenePredictionTask(Task):
    def __init__(
        self,
        model: torch.nn.Module,
        category: CategoryEncoder,
        prediction_type: str,
        scores: List[str],
    ):
        super().__init__()
        self.model = model

        self.label_to_idx = {
            str(category.decode(idx)): idx for idx in range(len(category))
        }
        self.idx_to_label = {
            idx: str(category.decode(idx)) for idx in range(len(category))
        }
        self.scores = [
            available_scores[score](label_to_idx=self.label_to_idx) for score in scores
        ]

        if prediction_type == "multilabel":
            self.activation: torch.nn.Module = torch.nn.Sigmoid()
            self.logit_loss = torch.nn.BCEWithLogitsLoss()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.Softmax(dim=-1)
            self.logit_loss = OneHotToCrossEntropyLoss()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def predict(self, x, x_len):
        logits, _ = self.model(x, x_len)
        prediction = self.activation(logits)
        return prediction, logits

    def forward(
        self, _mode: str, x, x_len, y, labels, unique_name: str, _dump_dir: str = None
    ):
        y_pr, y_hat = self.predict(x, x_len)
        loss = self.logit_loss(y_hat.float(), y.float())

        cacheable = dict(
            loss=loss.detach().cpu().item(),
            label=y.detach().cpu().unbind(dim=0),  # (batch_size, num_class)
            logit=y_hat.detach().cpu().unbind(dim=0),  # (batch_size, num_class)
            prediction=y_pr.detach().cpu().unbind(dim=0),  # (batch_size, num_class)
        )

        return loss, cacheable

    def log_scores(self, score_args):
        """Logs the metric score value for each score defined for the model"""
        assert hasattr(self, "scores"), "Scores for the model should be defined"
        end_scores = {}
        # The first score in the first `self.scores` is the optimization criterion
        for score in self.scores:
            score_ret = score(*score_args)
            validate_score_return_type(score_ret)
            # If the returned score is a tuple, store each subscore as separate entry
            if isinstance(score_ret, tuple):
                end_scores[f"{score}"] = score_ret[0][1]
                # All other scores will also be logged
                for (subscore, value) in score_ret:
                    end_scores[f"{score}_{subscore}"] = value
            elif isinstance(score_ret, float):
                end_scores[f"{score}"] = score_ret
            else:
                raise ValueError(
                    f"Return type {type(score_ret)} is unexpected. Return type of "
                    "the score function should either be a "
                    "tuple(tuple) or float."
                )
        return end_scores

    def reduction(
        self,
        _mode: str,
        cached_results: List[dict],
        _dump_dir: str,
    ):
        result = self.parse_cached_results(cached_results)

        target = torch.stack(result["label"], dim=0)
        prediction_logit = torch.stack(result["logit"], dim=0)
        prediction = torch.stack(result["prediction"], dim=0)

        loss = self.logit_loss(prediction_logit, target)

        logs = dict(
            loss=loss.detach().cpu().item(),
        )

        if _mode in ["valid", "test"]:
            logs.update(
                self.log_scores(
                    score_args=(
                        prediction.detach().cpu().numpy(),
                        target.detach().cpu().numpy(),
                    ),
                )
            )

        return logs
