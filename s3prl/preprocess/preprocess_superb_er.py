import os
import json
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

from s3prl.problem.common.superb_er import iemocap_for_superb

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("iemocap", help="The root path of IEMOCAP")
    parser.add_argument("test_fold", type=int)
    parser.add_argument(
        "--csv_dir",
        help="The directory for all output csvs",
        default="./data/superb_er",
    )
    parser.add_argument("--train_n_shot", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    fold_dir = csv_dir / f"fold_{args.test_fold}"
    iemocap_for_superb(fold_dir, fold_dir, args.iemocap, args.test_fold)

    os.rename(fold_dir / "valid.csv", fold_dir / "dev.csv")

    with (fold_dir / "fold_id").open("w") as f:
        json.dump(args.test_fold, f)

    emotion2utt = defaultdict(list)
    df = pd.read_csv(fold_dir / "train.csv")
    ids = df["id"].tolist()
    labels = df["label"].tolist()

    for idx, label in zip(ids, labels):
        emotion2utt[label].append(idx)

    if args.train_n_shot is not None:
        valid_utt = []
        random.seed(args.seed)

        for emotion, utts in emotion2utt.items():
            valid_utt.extend(random.sample(utts, k=args.train_n_shot))

        select = [idx in valid_utt for idx in df["id"]]
        selected_df = df[select]
        selected_df.to_csv(fold_dir / "train.csv", index=False)
