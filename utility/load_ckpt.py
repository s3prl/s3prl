import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

torch.load(args.path, map_location="cpu")
