import logging
import argparse
import random
from pathlib import Path
from collections import defaultdict, OrderedDict

import pandas as pd

from s3prl.dataio.corpus import VoxCeleb1SID

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("voxceleb1", help="The root path of VoxCeleb1")
    parser.add_argument(
        "--csv_dir",
        help="The directory for all output csvs",
        default="./data/superb_sid",
    )
    parser.add_argument("--train_n_shot", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus = VoxCeleb1SID(args.voxceleb1)
    train_data, valid_data, test_data = corpus.data_split

    if args.train_n_shot is not None:
        logger.info(f"Random sample {args.train_n_shot} utterance per speaker")

        utts = sorted(list(train_data.keys()))
        spk2utts = defaultdict(list)
        for utt in utts:
            spk = train_data[utt]["label"]
            spk2utts[spk].append(utt)
        spks = list(spk2utts.keys())
        spks.sort(key=lambda spk: len(spk2utts[spk]))
        sorted_spk2utts = OrderedDict()
        for spk in reversed(spks):
            sorted_spk2utts[spk] = spk2utts[spk]

        minimum_num_utt = len(sorted_spk2utts[list(sorted_spk2utts.keys())[0]])
        assert (
            args.train_n_shot <= minimum_num_utt
        ), f"The minimum number of utterance per speaker is {minimum_num_utt}"

        random.seed(args.seed)
        pruned_utts = []
        for spk, utts in sorted_spk2utts.items():
            num_pruned = len(utts) - args.train_n_shot
            assert num_pruned >= 0
            sampled_utts = random.sample(utts, k=num_pruned)
            pruned_utts.extend(sampled_utts)

        for utt in pruned_utts:
            del train_data[utt]

    def dict_to_csv(data_dict, csv_path):
        keys = sorted(list(data_dict.keys()))
        fields = sorted(data_dict[keys[0]].keys())
        data = dict()
        for field in fields:
            data[field] = []
            for key in keys:
                value = data_dict[key][field]
                if field == "label":
                    speaker_id = int(value.split("_")[-1])
                    value = f"id{speaker_id + 10001}"
                data[field].append(value)
        data["id"] = keys
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(exist_ok=True, parents=True)
    dict_to_csv(train_data, csv_dir / f"train.csv")
    dict_to_csv(valid_data, csv_dir / f"dev.csv")
    dict_to_csv(test_data, csv_dir / f"test.csv")
