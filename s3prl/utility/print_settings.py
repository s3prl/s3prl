import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ckpt", help="eg. result/downstream/ExpName/states-100.ckpt")
parser.add_argument("field_string", help="eg. config.runner.total_steps")
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
Args = ckpt["Args"]
Config = ckpt["Config"]

first_field, *remaining = args.field_string.split('.')
if first_field == 'args':
    assert len(remaining) == 1
    print(getattr(Args, remaining[0]))
elif first_field == 'config':
    target_config = Config
    for i, field_name in enumerate(remaining):
        if i == len(remaining) - 1:
            print(target_config[field_name])
        else:
            target_config = target_config[field_name]
