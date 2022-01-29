import random
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("official_split_file")
parser.add_argument("n_shot", type=int)
parser.add_argument("output")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
random.seed(args.seed)

spk2train = defaultdict(list)
spk2dev = defaultdict(list)
spk2test = defaultdict(list)
lines = Path(args.official_split_file).open().readlines()
for line in tqdm(lines, desc="reading official split file"):
    line = line.strip()
    split, uid = line.split()
    spk = uid.split("/")[0]
    if split == "1":
        spk2train[spk].append(uid)
    elif split == "2":
        spk2dev[spk].append(uid)
    elif split == "3":
        spk2test[spk].append(uid)
    else:
        raise ValueError

few_shot_spk2train = {}
for spk, uids in tqdm(spk2train.items(), desc="sampling few-shot"):
    uids = random.sample(uids, k=args.n_shot)
    few_shot_spk2train[spk] = uids

with Path(args.output).open("w") as file:
    for spk, uids in few_shot_spk2train.items():
        for uid in uids:
            print(1, uid, file=file)
    for spk, uids in spk2dev.items():
        for uid in uids:
            print(2, uid, file=file)
    for spk, uids in spk2test.items():
        for uid in uids:
            print(3, uid, file=file)
