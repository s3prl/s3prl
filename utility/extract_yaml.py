import sys
import yaml
import torch

ckpt_pth = sys.argv[1]
save_pth = sys.argv[2]

ckpt = torch.load(ckpt_pth, map_location='cpu')
config = ckpt['Settings']['Config']

with open(save_pth, 'w') as handle:
    yaml.dump(config, handle)
