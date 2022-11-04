import os
import torch
import random
import argparse
import numpy as np
from s3prl import hub
from s3prl.utility.helper import override

SAMPLE_RATE = 16000
BATCH_SIZE = 8

parser = argparse.ArgumentParser()
parser.add_argument("--upstream", "-u", required=True)
parser.add_argument("--pth", "-p", required=True)
parser.add_argument("--override", "-o")
parser.add_argument("--device", default="cuda")
parser.add_argument("--layer", "-l", type=int, default=-1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

config = {}
if args.override is not None:
    override(args.override, args, config)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

wavs = [
    torch.randn(random.randint(SAMPLE_RATE * 1, SAMPLE_RATE * 15)).to(args.device)
    for _ in range(BATCH_SIZE)
]

upstream = getattr(hub, args.upstream)(**config.get("upstream_expert", {})).to(args.device)
upstream.eval()

with torch.no_grad():
    hidden = upstream(wavs)["hidden_states"][args.layer].detach().cpu()
    torch.save(hidden, args.pth)
