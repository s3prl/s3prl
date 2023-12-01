import random
import argparse
from pathlib import Path


def write_metadata(data, filepath):
    with open(filepath, "w") as f:
        for d in data:
            print(d, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("all_metadata")
    parser.add_argument("train_metadata")
    parser.add_argument("valid_metadata")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    Path(args.train_metadata).parent.mkdir(exist_ok=True, parents=True)
    Path(args.valid_metadata).parent.mkdir(exist_ok=True, parents=True)

    with open(args.all_metadata) as f:
        lines = [line.strip() for line in f.readlines()]

    random.seed(args.seed)
    random.shuffle(lines)
    pivot = round(len(lines) * args.valid_ratio)
    valid_data = lines[:pivot]
    train_data = lines[pivot:]

    write_metadata(train_data, args.train_metadata)
    write_metadata(valid_data, args.valid_metadata)
