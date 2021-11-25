import sys
import torch

if len(sys.argv) != 3:
    print("Usage: python3 print_settings.py [ckpt] [config.runner.total_steps]")

ckpt = torch.load(sys.argv[1], map_location="cpu")
args = ckpt["Args"]
config = ckpt["Config"]

first_field, *remaining = sys.argv[2].split('.')
if first_field == 'args':
    assert len(remaining) == 1
    print(getattr(args, remaining[0]))
elif first_field == 'config':
    target_config = config
    for i, field_name in enumerate(remaining):
        if i == len(remaining) - 1:
            print(target_config[field_name])
        else:
            target_config = target_config[field_name]
