import sys
import torch
import matplotlib.pyplot as plt

ckpt_path = sys.argv[1]
imgname = sys.argv[2]

ckpt = torch.load(ckpt_path)
weights = ckpt['Classifier']['weight']
norm = weights.abs() / weights.abs().sum()
plt.plot(norm.cpu().numpy())
plt.savefig(imgname)

