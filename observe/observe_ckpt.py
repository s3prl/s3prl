import sys
import torch
from ipdb import set_trace

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path)
set_trace()

