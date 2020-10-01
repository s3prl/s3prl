import sys
import yaml
import torch
from argparse import Namespace

ckpt_pth = sys.argv[1]
args_pth = sys.argv[2]
config_pth = sys.argv[3]
new_ckpt_pth = sys.argv[4]

ckpt = torch.load(ckpt_pth, map_location='cpu')

# overwrite args
with open(args_pth, 'r') as handle:
    args = yaml.load(handle, Loader=yaml.FullLoader)
    ckpt['Settings']['Paras'] = Namespace(**args)

# overwrite config
with open(config_pth, 'r') as handle:
    config = yaml.load(handle, Loader=yaml.FullLoader)
    ckpt['Settings']['Config'] = config

torch.save(ckpt, new_ckpt_pth)
