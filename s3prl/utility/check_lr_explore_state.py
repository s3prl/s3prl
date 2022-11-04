import re
import sys
import argparse
from decimal import Decimal

from numpy import isin

parser = argparse.ArgumentParser()
parser.add_argument("--higher_better", action="store_true")
args = parser.parse_args()

lr2score = {}
for line in sys.stdin:
    line = line.strip()
    lr, score = line.split()
    score = float(score)
    lr2score[lr] = score
lr2score = [(lr, score) for lr, score in lr2score.items()]

def is_better(current, target):
    if args.higher_better and current > target:
        return True
    if not args.higher_better and current < target:
        return True
    return False

if len(lr2score) < 2:
    print("[status 5] Only a single lr/score pair is found. Please at least try two learning rates")

lr2score.sort(key=lambda x: float(x[0]))
both_end_best = []
small_end_best = None
large_end_best = None
for idx, (lr, score) in enumerate(lr2score):
    if idx == 0 and is_better(score, lr2score[idx + 1][1]):
        small_end_best = (lr, score)
    elif idx == (len(lr2score) - 1) and is_better(score, lr2score[idx - 1][1]):
        large_end_best = (lr, score)
    elif is_better(score, lr2score[idx - 1][1]) and is_better(score, lr2score[idx + 1][1]):
        both_end_best.append((lr, score))

error_message = (
    "This is not expected. "
    "Please open an issue and report the exploration result. "
    f"{', '.join([f'lr {lr} {score}' for lr, score in lr2score])}"
)
assert len(both_end_best) > 0 or small_end_best is not None or large_end_best is not None, error_message

def format_number(number):
    if isinstance(number, (int, float)):
        number = str(number)
    assert isinstance(number, str)
    return '%.2E' % Decimal(number)

if len(both_end_best) > 0:
    both_end_best.sort(key=lambda x: x[1], reverse=args.higher_better)
    best_lr, best_score = both_end_best[0]
    best_lr = float(best_lr)
    print("[status 0] Best lr/score found:", best_lr, best_score)
elif small_end_best is not None:
    best_lr, best_score = small_end_best
    best_lr = float(best_lr)
    print(
        f"[status 1] Please further explore smaller learning rates. "
        f"Suggest: {format_number(best_lr / 10)} and {format_number(best_lr / 100)}"
    )
elif large_end_best is not None:
    best_lr, best_score = large_end_best
    best_lr = float(best_lr)
    print(
        f"[status 1] Please further explore larger learning rates. "
        f"Suggest: {format_number(best_lr * 10)} and {format_number(best_lr * 100)}"
    )
else:
    print(f"[status 4] {error_message}")