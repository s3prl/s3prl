import random
import argparse
import pandas as pd
from pathlib import Path

from s3prl.dataio.corpus import LibriSpeech

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("librispeech", help="The root path of the LibriSpeech dataset")
    parser.add_argument(
        "--csv_dir",
        help="The directory for all the output csvs",
        default="./data/superb_pr",
    )
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus = LibriSpeech(args.librispeech)
    train_data, valid_data, test_data = corpus.data_split

    if args.train_ratio is not None:
        random.seed(args.seed)
        train_utts = sorted(list(train_data.keys()))
        train_utts = random.sample(
            train_utts, k=round(len(train_utts) * args.train_ratio)
        )
        train_data = {utt: train_data[utt] for utt in train_utts}

    def dict_to_csv(data_dict, csv_path):
        keys = sorted(list(data_dict.keys()))
        fields = sorted(data_dict[keys[0]].keys())
        data = dict()
        for field in fields:
            data[field] = []
            for key in keys:
                data[field].append(data_dict[key][field])
        data["id"] = keys
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(exist_ok=True, parents=True)
    dict_to_csv(train_data, csv_dir / "train.csv")
    dict_to_csv(valid_data, csv_dir / "dev.csv")
    dict_to_csv(test_data, csv_dir / "test.csv")
