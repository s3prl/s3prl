import sys
import yaml
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_pth', required=True)
parser.add_argument('--settings_dir', required=True)
parser.add_argument('--new_ckpt_pth', required=True)
paras = parser.parse_args()

ckpt = torch.load(paras.ckpt_pth, map_location='cpu')

with open(f'{paras.settings_dir}/args.yaml', 'r') as handle:
    args = yaml.load(handle, Loader=yaml.FullLoader)
    ckpt['Args'] = argparse.Namespace(**args)

with open(f'{paras.settings_dir}/config.yaml', 'r') as handle:
    config = yaml.load(handle, Loader=yaml.FullLoader)
    ckpt['Config'] = config

torch.save(ckpt, paras.new_ckpt_pth)
