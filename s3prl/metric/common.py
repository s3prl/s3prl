"""
Commonly used metrics

Authors
  * Leo 2022
  * Heng-Jui Chang 2022
  * Haibin Wu 2022
"""

from typing import List, Union

import editdistance as ed
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, roc_curve

__all__ = [
    "accuracy",
    "ter",
    "wer",
    "per",
    "cer",
    "compute_eer",
    "compute_minDCF",
]


def accuracy(xs, ys, item_same_fn=None):
    if isinstance(xs, (tuple, list)):
        assert isinstance(ys, (tuple, list))
        return _accuracy_impl(xs, ys, item_same_fn)
    elif isinstance(xs, dict):
        assert isinstance(ys, dict)
        keys = sorted(list(xs.keys()))
        xs = [xs[k] for k in keys]
        ys = [ys[k] for k in keys]
        return _accuracy_impl(xs, ys, item_same_fn)
    else:
        raise ValueError


def _accuracy_impl(xs, ys, item_same_fn=None):
    item_same_fn = item_same_fn or (lambda x, y: x == y)
    same = [int(item_same_fn(x, y)) for x, y in zip(xs, ys)]
    return sum(same) / len(same)


def ter(hyps: List[Union[str, List[str]]], refs: List[Union[str, List[str]]]) -> float:
    """Token error rate calculator.

    Args:
        hyps (List[Union[str, List[str]]]): List of hypotheses.
        refs (List[Union[str, List[str]]]): List of references.

    Returns:
        float: Averaged token error rate overall utterances.
    """
    error_tokens = 0
    total_tokens = 0
    for h, r in zip(hyps, refs):
        error_tokens += ed.eval(h, r)
        total_tokens += len(r)
    return float(error_tokens) / float(total_tokens)


def wer(hyps: List[str], refs: List[str]) -> float:
    """Word error rate calculator.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.

    Returns:
        float: Averaged word error rate overall utterances.
    """
    hyps = [h.split(" ") for h in hyps]
    refs = [r.split(" ") for r in refs]
    return ter(hyps, refs)


def per(hyps: List[str], refs: List[str]) -> float:
    """Phoneme error rate calculator.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.

    Returns:
        float: Averaged phoneme error rate overall utterances.
    """
    return wer(hyps, refs)


def cer(hyps: List[str], refs: List[str]) -> float:
    """Character error rate calculator.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.

    Returns:
        float: Averaged character error rate overall utterances.
    """
    return ter(hyps, refs)


def compute_eer(labels: List[int], scores: List[float]):
    """Compute equal error rate.

    Args:
        scores (List[float]): List of hypotheses.
        labels (List[int]): List of references.

    Returns:
        eer (float): Equal error rate.
        treshold (float): The treshold to accept a target trial.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold


def compute_minDCF(
    labels: List[int],
    scores: List[float],
    p_target: float = 0.01,
    c_miss: int = 1,
    c_fa: int = 1,
):
    """Compute MinDCF.
    Computes the minimum of the detection cost function.  The comments refer to
    equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.

    Args:
        scores (List[float]): List of hypotheses.
        labels (List[int]): List of references.
        p (float): The prior probability of positive class.
        c_miss (int): The cost of miss.
        c_fa (int): The cost of false alarm.

    Returns:
        min_dcf (float): The calculated min_dcf.
        min_c_det_threshold (float): The treshold to calculate min_dcf.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnr)):
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
