import argparse
from pathlib import Path

import torch

from s3prl import Dataset, Output, Task
from s3prl.base.object import Object

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "load_from", help="The directory containing all the checkpoints"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    load_from = Path(args.load_from)

    task: Task = Object.load_checkpoint(load_from / "task.ckpt").to(device)
    task.eval()

    test_dataset: Dataset = Object.load_checkpoint(load_from / "test_dataset.ckpt")
    test_dataloader = test_dataset.to_dataloader(batch_size=1, num_workers=6)

    with torch.no_grad():
        for batch in test_dataloader:
            batch: Output = batch.to(device)
            result = task(**batch.subset("x", "x_len", as_type="dict"))
            for name, prediction in zip(batch.name, result.prediction):
                print(name, prediction)


if __name__ == "__main__":
    main()
