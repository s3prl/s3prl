import sys
import yaml
import torch

ckpt_pth = sys.argv[1]
save_dir = sys.argv[2]

ckpt = torch.load(ckpt_pth, map_location='cpu')

with open(f'{save_dir}/args.yaml', 'w') as handle:
    args = ckpt['Settings']['Paras']
    yaml.dump(vars(args), handle)

with open(f'{save_dir}/config.yaml', 'w') as handle:
    config = ckpt['Settings']['Config']
    yaml.dump(config, handle)
