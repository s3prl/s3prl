import argparse

import yaml
import torch
from omegaconf import OmegaConf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fairseq_ckpt")
    parser.add_argument("output_filepath")
    args = parser.parse_args()

    ckpt = torch.load(args.fairseq_ckpt, map_location="cpu")
    conf = OmegaConf.create(ckpt["cfg"])
    conf = OmegaConf.to_container(conf, resolve=True)

    with open(args.output_filepath, "w") as f:
        yaml.dump(conf, f)
