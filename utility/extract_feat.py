import logging
import argparse
from pathlib import Path

import torch

from s3prl.nn import S3PRLUpstream
from s3prl.util.pseudo_data import get_pseudo_wavs
from s3prl.util.override import parse_overrides

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--output_dir", default="./sample_hidden_states")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--device", default="cuda")
    args, others = parser.parse_known_args()

    overrides = parse_overrides(others)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model = S3PRLUpstream(args.name, refresh=args.refresh, extra_conf=overrides).to(
        args.device
    )
    model.eval()

    with torch.no_grad():
        x, x_len = get_pseudo_wavs(padded=True)
        hs, hs_len = model(x.to(args.device), x_len.to(args.device))
        hs = [h.detach().cpu() for h, h_len in zip(hs, hs_len)]

    torch.save(hs, output_dir / f"{args.name}.pt")
