import logging
import argparse
from typing import Tuple, List, Dict

import numpy as np
from scipy.stats import ttest_rel
from mlxtend.evaluate import mcnemar_table, mcnemar

from s3prl.metric import per, wer, cer, accuracy

logger = logging.getLogger(__name__)


def read_file(filepath: str) -> Dict[str, float]:
    name2value = {}
    with open(filepath) as f:
        for line in f.readlines():
            line = line.strip()
            name, value = line.split(maxsplit=1)
            assert name not in name2value
            name2value[name] = value
    return name2value


def form_pairs(
    truths: Dict[str, float], predicts1: Dict[str, float], predicts2: Dict[str, float]
) -> List[Tuple[str, str, str]]:
    names = sorted(truths.keys())
    return [(truths[name], predicts1[name], predicts2[name]) for name in names]


def per_sample_metrics(pairs: List[tuple], per_metric_fn: callable):
    metrics1, metrics2 = [], []
    for gt, pred1, pred2 in pairs:
        metrics1.append(per_metric_fn(pred1, gt))
        metrics2.append(per_metric_fn(pred2, gt))
    return metrics1, metrics2


def pairwise_t_test_pvalue(metrics1: List[float], metrics2: List[float]) -> float:
    stats = ttest_rel(metrics1, metrics2, nan_policy="raise")
    return stats.pvalue


def mcnemar_test_pvalue(metrics1: List[bool], metrics2: List[bool]) -> float:
    target = np.array([1] * len(metrics1)).astype(np.int32)
    correct1 = np.array(metrics1).astype(np.int32)
    correct2 = np.array(metrics2).astype(np.int32)
    table = mcnemar_table(target, correct1, correct2)
    sample_size = table[0, 1] + table[1, 0]
    if sample_size < 25:
        logger.warning(
            "sample size < 25, compute the exact p-value from the binomial distribution"
        )
        chi2, p = mcnemar(table, exact=True)
    else:
        chi2, p = mcnemar(table, corrected=True)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("metric", choices=["per", "cer", "wer", "acc"])
    parser.add_argument("ground_truth", help="The ground truth file")
    parser.add_argument("prediction1", help="The 1st prediction file")
    parser.add_argument("prediction2", help="The 2nd prediction file")
    args = parser.parse_args()

    if args.metric in ["per", "cer", "wer"]:
        groundtruth = read_file(args.ground_truth)
        prediction1 = read_file(args.prediction1)
        prediction2 = read_file(args.prediction2)
        pairs = form_pairs(groundtruth, prediction1, prediction2)
        gs, p1, p2 = zip(*pairs)

        if args.metric == "per":
            metric_fn = per
        elif args.metric == "cer":
            metric_fn = cer
        elif args.metric == "wer":
            metric_fn = wer

        macro_metric1 = metric_fn(p1, gs)
        macro_metric2 = metric_fn(p2, gs)

        metrics1, metrics2 = per_sample_metrics(
            pairs, lambda pred, gt: float(metric_fn([pred], [gt]))
        )
        pvalue = pairwise_t_test_pvalue(metrics1, metrics2)


    elif args.metric == "acc":
        groundtruth = read_file(args.ground_truth)
        prediction1 = read_file(args.prediction1)
        prediction2 = read_file(args.prediction2)
        pairs = form_pairs(groundtruth, prediction1, prediction2)
        gs, p1, p2 = zip(*pairs)

        macro_metric1 = accuracy(p1, gs)
        macro_metric2 = accuracy(p2, gs)

        metrics1, metrics2 = per_sample_metrics(pairs, lambda pred, gt: pred == gt)
        pvalue = mcnemar_test_pvalue(metrics1, metrics2)

    print(f"{args.metric} 1: {macro_metric1}")
    print(f"{args.metric} 2: {macro_metric2}")
    print(f"pvalue: {pvalue}")
