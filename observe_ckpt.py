import sys
import torch
from utils import timer
from ipdb import set_trace

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path)
set_trace()

