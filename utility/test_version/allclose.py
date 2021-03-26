import sys
import torch

pth1 = torch.load(sys.argv[1])
pth2 = torch.load(sys.argv[2])

print((pth1 - pth2).abs().max().item())

