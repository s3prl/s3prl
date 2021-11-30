# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ find_best_epoch.py ]
#   Synopsis     [ Script to read and find best epoch ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""

import argparse
from io import open
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--upstream", type=str, required=True, help="upstream")
    parser.add_argument("--task", type=str, required=True, help="task")
    parser.add_argument("--tag", type=str, required=True, help="tag")
    parser.add_argument("--vocoder", type=str, required=True, help="vocoder name")
    parser.add_argument("--expdir", type=str, default="../../result/downstream", help="expdir")
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--start_epoch", default=10000, type=int)
    parser.add_argument("--end_epoch", default=50000, type=int)
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

    scores = []
    for ep in epochs:
        log_file = os.path.join(args.expdir, f"a2a_vc_vctk_{args.tag}_{args.upstream}", str(ep), f"{args.vocoder}_wav", f"obj_{args.num_samples}samples.log")
        if args.task == "task1":
            result = grep(log_file, "Mean")[0].split("Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER, accept rate: ")[1].split(" ")
        elif args.task == "task2":
            result = grep(log_file, "Mean")[0].split("Mean CER, WER, accept rate: ")[1].split(" ")
        scores.append([str(ep)] + result)
    best = min(scores, key=lambda x: -float(x[-1]))
        
    print(f"{args.upstream} {args.num_samples} samples epoch {best[0]} best:", " ".join(best[1:4]), " ".join(best[5:]),)
