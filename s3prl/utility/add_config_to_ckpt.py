import yaml
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--out_ckpt', required=True)
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location='cpu')
with open(args.config, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

ckpt['config'] = config
torch.save(ckpt, args.out_ckpt)

