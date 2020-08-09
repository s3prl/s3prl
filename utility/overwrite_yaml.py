import sys
import yaml
import torch

ckpt_pth = sys.argv[1]
config_pth = sys.argv[2]
new_ckpt_pth = sys.argv[3]

ckpt = torch.load(ckpt_pth, map_location='cpu')
with open(config_pth, 'r') as handle:
    config = yaml.load(config_pth, Loader=yaml.FullLoader)

ckpt['Settings']['Config'] = config
torch.save(ckpt, new_ckpt_pth)
