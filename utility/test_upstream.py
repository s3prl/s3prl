import random
import argparse
import numpy as np

import torch
from s3prl.nn import S3PRLUpstream
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 16000
BATCH_SIZE = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream")
    parser.add_argument("--ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    upstream = S3PRLUpstream(args.upstream, args.ckpt).to(args.device)
    wavs = [
        torch.randn(random.randint(SAMPLE_RATE * 1, SAMPLE_RATE * 15)).to(args.device)
        for _ in range(BATCH_SIZE)
    ]
    wavs_len = torch.LongTensor([len(w) for w in wavs]).to(args.device)
    wavs = pad_sequence(wavs, batch_first=True)

    with torch.no_grad():
        upstream.eval()
        hidden, hidden_len = upstream(wavs, wavs_len)
