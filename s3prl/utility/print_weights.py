import argparse

import torch


parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
weights = ckpt["Featurizer"]["weights"]
norm_weights = torch.nn.functional.softmax(weights)
for weight in norm_weights.tolist():
    print(weight)

