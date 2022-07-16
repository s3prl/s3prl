import torch
import argparse
from pathlib import Path

from s3prl import hub
from s3prl.util.pseudo_data import get_pseudo_wavs

SAMPLE_RATE = 16000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--output_dir", default="../sample_hidden_states")
    parser.add_argument("--ckpt")
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model = getattr(hub, args.name)(ckpt=args.ckpt, legacy=args.legacy)
    model.eval()

    with torch.no_grad():
        hidden_states = model(get_pseudo_wavs())["hidden_states"]
        hs = [h.detach().cpu() for h in hidden_states]

    torch.save(hs, output_dir / f"{args.name}.pt")
