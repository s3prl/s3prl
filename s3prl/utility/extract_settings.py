import os
import sys
import yaml
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_pth', required=True)
parser.add_argument('--save_dir', required=True)
paras = parser.parse_args()

ckpt = torch.load(paras.ckpt_pth, map_location='cpu')
os.makedirs(paras.save_dir, exist_ok=True)

with open(f'{paras.save_dir}/args.yaml', 'w') as handle:
    args = ckpt['Args']
    yaml.dump(vars(args), handle)

with open(f'{paras.save_dir}/config.yaml', 'w') as handle:
    config = ckpt['Config']
    yaml.dump(config, handle)
