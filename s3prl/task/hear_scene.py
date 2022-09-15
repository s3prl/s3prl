import torch
from typing import List, Dict, Any

from s3prl import Logs, Output
from s3prl.nn.upstream import SAMPLE_RATE
from s3prl.task.base import Task
from s3prl.encoder.category import CategoryEncoder
from s3prl.metric.hear import (
    available_scores,
    validate_score_return_type,
)


class OneHotToCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.all(torch.sum(y, dim=1) == y.new_ones(y.shape[0]))
        y = y.argmax(dim=1)
        return self.loss(y_hat, y)


class HearScenePredictionTask(Task):
    """
    Prediction model with simple scoring over entire audio scenes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        category: CategoryEncoder,
        prediction_type: str,
        scores: List[str],
        **kwds,
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

    def forward(self, split, x, x_len, y, labels, **kwds):
        y_pr, y_hat = self.predict(x, x_len)
        loss = self.logit_loss(y_hat.float(), y.float())

        return Output(
            loss=loss,
            label=y,  # (batch_size, num_class)
            logit=y_hat,  # (batch_size, num_class)
            prediction=y_pr,  # (batch_size, num_class)
        )

    def log_scores(self, score_args, logs: Logs):
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

        for score_name in end_scores:
            logs.add_scalar(score_name, end_scores[score_name])

        return logs

    def reduction(
        self, split: str, batch_results: List, on_epoch_end: bool = None, **kwds
    ):
        target, prediction, prediction_logit = [], [], []
        for batch in batch_results:
            target.append(batch["label"])
            prediction.append(batch["prediction"])
            prediction_logit.append(batch["logit"])

        target = torch.cat(target, dim=0)
        prediction = torch.cat(prediction, dim=0)
        prediction_logit = torch.cat(prediction_logit, dim=0)

        logs = Logs()
        logs.add_scalar(f"loss", self.logit_loss(prediction_logit, target))

        if split in ["valid", "test"]:
            logs = self.log_scores(
                score_args=(
                    prediction.detach().cpu().numpy(),
                    target.detach().cpu().numpy(),
                ),
                logs=logs,
            )

        return Output(
            logs=logs,
        )
