import sys
import yaml
import torch

ckpt_pth = sys.argv[1]
save_dir = sys.argv[2]

ckpt = torch.load(ckpt_pth, map_location='cpu')

with open(f'{save_dir}/args.tmp.yaml', 'w') as handle:
    args = ckpt['Settings']['Paras']
    yaml.dump(vars(args), handle)

with open(f'{save_dir}/config.tmp.yaml', 'w') as handle:
    config = ckpt['Settings']['Config']
    yaml.dump(config, handle)

with open(f'{save_dir}/global_step.tmp.txt', 'w') as handle:
    step = ckpt['Global_step']
    handle.write(f'{step}\n')
