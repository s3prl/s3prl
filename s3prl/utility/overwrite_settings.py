from pathlib import Path

import yaml
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('settings_dir')
parser.add_argument('ckpt_path')
parser.add_argument('new_ckpt_path')
paras = parser.parse_args()

ckpt = torch.load(paras.ckpt_path, map_location='cpu')

with open(f'{paras.settings_dir}/args.yaml', 'r') as handle:
    args = yaml.load(handle, Loader=yaml.FullLoader)
    ckpt['Args'] = argparse.Namespace(**args)

with open(f'{paras.settings_dir}/config.yaml', 'r') as handle:
    config = yaml.load(handle, Loader=yaml.FullLoader)
    ckpt['Config'] = config

Path(paras.new_ckpt_path).parent.mkdir(exist_ok=True, parents=True)
torch.save(ckpt, paras.new_ckpt_path)
