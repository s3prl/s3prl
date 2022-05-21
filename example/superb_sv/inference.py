import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from s3prl import Output
from s3prl.base.object import Object
from s3prl.dataset import Dataset
from s3prl.task import Task

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_from",
        type=str,
        default="result/sv",
        help="The directory containing all the checkpoints",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    load_from = Path(args.load_from)

    task: Task = Object.load_checkpoint(load_from / "task.ckpt").to(device)
    task.eval()

    test_dataset: Dataset = Object.load_checkpoint(load_from / "test_dataset.ckpt")
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=6, collate_fn=test_dataset.collate_fn
    )

    with torch.no_grad():
        for batch in test_dataloader:
            batch: Output = batch.to(device)
            result = task(**batch.subset("x", "x_len", as_type="dict"))
            print(result.hidden_states.shape)


if __name__ == "__main__":
    main()
