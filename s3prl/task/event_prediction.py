# Copyright Hear Benchmark Team
# Copyright Shu-wen Yang (refactor from https://github.com/hearbenchmark/hear-eval-kit)

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import more_itertools
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from s3prl.dataio.encoder.category import CategoryEncoder
from s3prl.task.base import Task

from ._hear_score import available_scores, validate_score_return_type

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

__all__ = ["EventPredictionTask"]


def create_events_from_prediction(
    prediction_dict: Dict[float, torch.Tensor],
    idx_to_label: Dict[int, str],
    threshold: float = 0.5,
    median_filter_ms: float = 150,
    min_duration: float = 60.0,
) -> List[Dict[str, Union[float, str]]]:
    """
    Takes a set of prediction tensors keyed on timestamps and generates events.
    (This is for one particular audio scene.)
    We convert the prediction tensor to a binary label based on the threshold value. Any
    events occurring at adjacent timestamps are considered to be part of the same event.
    This loops through and creates events for each label class.
    We optionally apply median filtering to predictions.
    We disregard events that are less than the min_duration milliseconds.

    Args:
        prediction_dict: A dictionary of predictions keyed on timestamp
            {timestamp -> prediction}. The prediction is a tensor of label
            probabilities.
        idx_to_label: Index to label mapping.
        threshold: Threshold for determining whether to apply a label
        min_duration: the minimum duration in milliseconds for an
                event to be included.

    Returns:
        A list of dicts withs keys "label", "start", and "end"
    """
    # Make sure the timestamps are in the correct order
    timestamps = np.array(sorted(prediction_dict.keys()))

    # Create a sorted numpy matrix of frame level predictions for this file. We convert
    # to a numpy array here before applying a median filter.
    predictions = np.stack(
        [prediction_dict[t].detach().cpu().numpy() for t in timestamps]
    )

    # Optionally apply a median filter here to smooth out events.
    ts_diff = np.mean(np.diff(timestamps))
    if median_filter_ms:
        filter_width = int(round(median_filter_ms / ts_diff))
        if filter_width:
            predictions = median_filter(predictions, size=(filter_width, 1))

    # Convert probabilities to binary vectors based on threshold
    predictions = (predictions > threshold).astype(np.int8)

    events = []
    for label in range(predictions.shape[1]):
        for group in more_itertools.consecutive_groups(
            np.where(predictions[:, label])[0]
        ):
            grouptuple = tuple(group)
            assert (
                tuple(sorted(grouptuple)) == grouptuple
            ), f"{sorted(grouptuple)} != {grouptuple}"
            startidx, endidx = (grouptuple[0], grouptuple[-1])

            start = timestamps[startidx]
            end = timestamps[endidx]
            # Add event if greater than the minimum duration threshold
            if end - start >= min_duration:
                events.append(
                    {"label": idx_to_label[label], "start": start, "end": end}
                )

    # This is just for pretty output, not really necessary
    events.sort(key=lambda k: k["start"])
    return events


def get_events_for_all_files(
    predictions: torch.Tensor,
    filenames: List[str],
    timestamps: torch.Tensor,
    idx_to_label: Dict[int, str],
    postprocessing_grid: Dict[str, List[float]],
    postprocessing: Optional[Tuple[Tuple[str, Any], ...]] = None,
) -> Dict[Tuple[Tuple[str, Any], ...], Dict[str, List[Dict[str, Union[str, float]]]]]:
    """
    Produces lists of events from a set of frame based label probabilities.
    The input prediction tensor may contain frame predictions from a set of different
    files concatenated together. file_timestamps has a list of filenames and
    timestamps for each frame in the predictions tensor.

    We split the predictions into separate tensors based on the filename and compute
    events based on those individually.

    If no postprocessing is specified (during training), we try a
    variety of ways of postprocessing the predictions into events,
    from the postprocessing_grid including median filtering and
    minimum event length.

    If postprocessing is specified (during test, chosen at the best
    validation epoch), we use this postprocessing.

    Args:
        predictions: a tensor of frame based multi-label predictions.
        filenames: a list of filenames where each entry corresponds
            to a frame in the predictions tensor.
        timestamps: a list of timestamps where each entry corresponds
            to a frame in the predictions tensor.
        idx_to_label: Index to label mapping.
        postprocessing: See above.

    Returns:
        A dictionary from filtering params to the following values:
        A dictionary of lists of events keyed on the filename slug.
        The event list is of dicts of the following format:
            {"label": str, "start": float ms, "end": float ms}
    """
    # This probably could be more efficient if we make the assumption that
    # timestamps are in sorted order. But this makes sure of it.
    assert predictions.shape[0] == len(filenames)
    assert predictions.shape[0] == len(timestamps)
    event_files: Dict[str, Dict[float, torch.Tensor]] = {}
    for i, (filename, timestamp) in enumerate(zip(filenames, timestamps)):
        slug = Path(filename).name

        # Key on the slug to be consistent with the ground truth
        if slug not in event_files:
            event_files[slug] = {}

        # Save the predictions for the file keyed on the timestamp
        event_files[slug][float(timestamp) * 1000] = predictions[i]

    # Create events for all the different files. Store all the events as a dictionary
    # with the same format as the ground truth from the luigi pipeline.
    # Ex) { slug -> [{"label" : "woof", "start": 0.0, "end": 2.32}, ...], ...}
    event_dict: Dict[
        Tuple[Tuple[str, Any], ...], Dict[str, List[Dict[str, Union[float, str]]]]
    ] = {}
    if postprocessing:
        postprocess = postprocessing
        event_dict[postprocess] = {}
        logger.info("Use searched postprocess config to decode")
        for slug, timestamp_predictions in event_files.items():
            event_dict[postprocess][slug] = create_events_from_prediction(
                timestamp_predictions, idx_to_label, **dict(postprocess)
            )
    else:
        postprocessing_confs = list(ParameterGrid(postprocessing_grid))
        for postprocess_dict in tqdm(
            postprocessing_confs, desc="Search postprocessing"
        ):
            postprocess = tuple(postprocess_dict.items())
            event_dict[postprocess] = {}
            for slug, timestamp_predictions in event_files.items():
                event_dict[postprocess][slug] = create_events_from_prediction(
                    timestamp_predictions, idx_to_label, **postprocess_dict
                )

    return event_dict


def label_vocab_nlabels(embedding_path: Path) -> Tuple[pd.DataFrame, int]:
    label_vocab = pd.read_csv(embedding_path.joinpath("labelvocabulary.csv"))

    nlabels = len(label_vocab)
    assert nlabels == label_vocab["idx"].max() + 1
    return (label_vocab, nlabels)


class OneHotToCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.all(
            torch.sum(y, dim=1) == torch.ones(y.shape[0], device=self.device)
        )
        y = y.argmax(dim=1)
        return self.loss(y_hat, y)


class EventPredictionTask(Task):
    def __init__(
        self,
        model: torch.nn.Module,
        category: CategoryEncoder,
        prediction_type: str,
        scores: List[str],
        postprocessing_grid: Dict[str, List[float]],
        valid_target_events: Dict[str, List[Dict[str, Any]]] = None,
        test_target_events: Dict[str, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.model = model
        assert isinstance(self.model.downsample_rate, int)
        self.feat_frame_shift = self.model.downsample_rate

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
            self.activation = torch.nn.Softmax()
            self.logit_loss = OneHotToCrossEntropyLoss()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

        self.target_events = {
            "valid": valid_target_events,
            "test": test_target_events,
        }
        # For each epoch, what postprocessing parameters were best
        self.postprocessing_grid = postprocessing_grid
        self.best_postprocessing = None

    def get_state(self):
        return {
            "best_postprocessing": self.best_postprocessing,
        }

    def set_state(self, state: dict):
        self.best_postprocessing = state["best_postprocessing"]

    def predict(self, x, x_len):
        logits, _ = self.model(x, x_len)
        prediction = self.activation(logits)
        return prediction, logits, x_len

    def _match_length(self, inputs, labels):
        """
        Since the upstream extraction process can sometimes cause a mismatch
        between the seq lenth of inputs and labels:
        - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
        - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
        Note that the length of labels should never be changed.
        """
        input_len, label_len = inputs.size(1), labels.size(1)

        factor = int(round(label_len / input_len))
        assert factor == 1
        if input_len > label_len:
            inputs = inputs[:, :label_len, :]
        elif input_len < label_len:
            pad_vec = inputs[:, -1, :].unsqueeze(1)  # (batch_size, 1, feature_dim)
            inputs = torch.cat(
                (inputs, pad_vec.repeat(1, label_len - input_len, 1)), dim=1
            )  # (batch_size, seq_len, feature_dim), where seq_len == labels.size(-1)
        return inputs

    def train_step(
        self, x, x_len, y, y_len, record_id, chunk_id, _dump_dir: str = None
    ):
        y_hat, y_hat_len = self.model(x, x_len)
        y_hat = self._match_length(y_hat, y)

        assert y_hat.size(-1) == y.size(-1), f"{y_hat.size(-1)} == {y.size(-1)}"

        hidden_size = y_hat.size(-1)
        loss = self.logit_loss(
            y_hat.reshape(-1, hidden_size).float(), y.reshape(-1, hidden_size).float()
        )

        cacheable = {
            "loss": loss.detach().cpu().item(),
        }

        return loss, cacheable

    def train_reduction(self, batch_results: list, _dump_dir: str = None):
        loss = []
        for batch in batch_results:
            loss.append(batch["loss"])
        loss = torch.FloatTensor(loss).mean().item()

        return {
            "loss": loss.detach().cpu().item(),
        }

    def _eval_step(
        self,
        x,
        x_len,
        y,
        y_len,
        record_id: List[str],
        chunk_id: List[int],
        _dump_dir: str = None,
    ):
        y_pr, y_hat, y_pr_len = self.predict(x, x_len)
        y_pr = self._match_length(y_pr, y)
        y_hat = self._match_length(y_hat, y)

        assert len(set(record_id)) == 1
        chunk_id = chunk_id.detach().cpu().tolist()
        assert sorted(chunk_id) == chunk_id

        y_pr_trim, y_hat_trim, y_trim = [], [], []
        for _p, _h, _y, length in zip(y_pr, y_hat, y, y_len):
            y_pr_trim.append(_p[:length])
            y_hat_trim.append(_h[:length])
            y_trim.append(_y[:length])
        y_pr_trim = torch.cat(y_pr_trim, dim=0)
        y_hat_trim = torch.cat(y_hat_trim, dim=0)
        y_trim = torch.cat(y_trim, dim=0)

        return 0, dict(
            label=y_trim,  # (seqlen, num_class)
            logit=y_hat_trim,  # (seqlen, num_class)
            prediction=y_pr_trim,  # (seqlen, num_class)
            record_id=record_id[0],  # List[str]
        )

    def valid_step(self, *args, **kwds):
        return self._eval_step(*args, **kwds)

    def test_step(self, *args, **kwds):
        return self._eval_step(*args, **kwds)

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

    def valid_reduction(self, cached_results: list, _dump_dir: str = None):
        return self.eval_reduction("valid", cached_results, _dump_dir)

    def test_reduction(self, cached_results: list, _dump_dir: str = None):
        return self.eval_reduction("test", cached_results, _dump_dir)

    def eval_reduction(self, _mode: str, cached_results: list, _dump_dir: str = None):
        target, prediction, prediction_logit, filename, timestamp = [], [], [], [], []
        for batch in cached_results:
            length = batch["label"].size(0)
            assert batch["prediction"].size(0) == length
            assert batch["logit"].size(0) == length

            target.append(batch["label"])
            prediction.append(batch["prediction"])
            prediction_logit.append(batch["logit"])
            filename += [batch["record_id"]] * length

            ts = (
                torch.arange(1, length + 1).float() * self.feat_frame_shift
                - self.feat_frame_shift / 2
            ) / SAMPLE_RATE
            timestamp += ts.tolist()

        target = torch.cat(target, dim=0)  # (timestamp, hidden_size)
        prediction = torch.cat(prediction, dim=0)
        prediction_logit = torch.cat(prediction_logit, dim=0)
        timestamp = torch.FloatTensor(timestamp)

        loss = self.logit_loss(prediction_logit.float(), target.float())

        logs = {"loss": loss.detach().cpu().item()}

        if _mode in ["valid", "test"]:
            # events in miniseconds
            predicted_events_by_postprocessing = get_events_for_all_files(
                prediction,
                filename,
                timestamp,
                self.idx_to_label,
                self.postprocessing_grid,
                self.best_postprocessing if _mode == "test" else None,
            )

            score_and_postprocessing = []
            for postprocessing in tqdm(predicted_events_by_postprocessing):
                predicted_events = predicted_events_by_postprocessing[postprocessing]
                primary_score_fn = self.scores[0]
                primary_score_ret = primary_score_fn(
                    predicted_events, self.target_events[_mode]
                )
                if isinstance(primary_score_ret, tuple):
                    primary_score = primary_score_ret[0][1]
                elif isinstance(primary_score_ret, float):
                    primary_score = primary_score_ret
                else:
                    raise ValueError(
                        f"Return type {type(primary_score_ret)} is unexpected. "
                        "Return type of the score function should either be a "
                        "tuple(tuple) or float. "
                    )
                if np.isnan(primary_score):
                    primary_score = 0.0
                score_and_postprocessing.append((primary_score, postprocessing))
            score_and_postprocessing.sort(reverse=True)

            if _mode in ["valid", "test"]:
                self.best_postprocessing = score_and_postprocessing[0][1]
                logger.info(f"Best postprocessing: {self.best_postprocessing}")

            predicted_events = predicted_events_by_postprocessing[
                self.best_postprocessing
            ]

            if _mode == "test":
                self.test_predictions = {
                    "target": target.detach().cpu(),
                    "prediction": prediction.detach().cpu(),
                    "prediction_logit": prediction_logit.detach().cpu(),
                    "target_events": self.target_events[_mode],
                    "predicted_events": predicted_events,
                    "timestamp": timestamp,
                }

            score_logs = self.log_scores(
                score_args=(predicted_events, self.target_events[_mode])
            )
            logs.update(score_logs)

        return logs
