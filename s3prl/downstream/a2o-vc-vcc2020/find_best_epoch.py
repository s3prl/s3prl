# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ find_best_epoch.py ]
#   Synopsis     [ Script to read and find best epoch (used with batch training mode only) ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import argparse
from io import open
import os

srcspks = ["SEF1", "SEF2", "SEM1", "SEM2"]
task1_trgspks = ["TEF1", "TEF2", "TEM1", "TEM2"]
task2_trgspks = ["TFF1", "TFM1", "TGF1", "TGM1", "TMF1", "TMM1"]


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--upstream", type=str, required=True, help="upstream")
    parser.add_argument("--task", type=str, required=True, choices=["task1", "task2"], help="task")
    parser.add_argument("--tag", type=str, required=True, help="tag")
    parser.add_argument("--vocoder", type=str, required=True, help="vocoder name")
    parser.add_argument("--expdir", type=str, default="result/downstream", help="expdir")
    parser.add_argument("--start_epoch", default=4000, type=int)
    parser.add_argument("--end_epoch", default=10000, type=int)
    parser.add_argument("--step_epoch", default=1000, type=int)
    parser.add_argument(
        "--out",
        "-O",
        type=str,
        help="The output filename. " "If omitted, then output to sys.stdout",
    )
    return parser


def grep(filepath, query):
    lines = []
    with open(filepath, "r") as f:
        for line in f:
            if query in line:
                lines.append(line.rstrip())
    return lines


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    epochs = list(range(args.start_epoch, args.end_epoch+args.step_epoch, args.step_epoch))
    bests = []
    trgspks = task1_trgspks if args.task == "task1" else task2_trgspks

    for trgspk in trgspks:
        scores = []
        for ep in epochs:
            # the expdir name pattern is consistent with the batch training script
            log_file = os.path.join(args.expdir, f"a2o_vc_vcc2020_{args.tag}_{trgspk}_{args.upstream}", str(ep), f"{args.vocoder}_wav", "obj.log")
            if args.task == "task1":
                result = grep(log_file, "Mean")[0].split("Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER, accept rate: ")[1].split(" ")
            elif args.task == "task2":
                result = grep(log_file, "Mean")[0].split("Mean CER, WER, accept rate: ")[1].split(" ")
            scores.append(result)
        
        # task 1: choose by MCD; task 2: choose by CER
        best = min(scores, key=lambda x: float(x[0]))
        bests.append(best)

    if args.task == "task1":
        avg = [f"{(sum([float(best[i]) for best in bests]) / 4.0):.2f}" for i in range(7)]
    elif args.task == "task2":
        avg = [f"{(sum([float(best[i]) for best in bests]) / 6.0):.2f}" for i in range(3)]
    print("Best result:"+" ".join(avg))
