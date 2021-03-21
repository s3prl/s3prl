import os
import sys
import torch
import torchaudio

filepath = sys.argv[1]
wav, sr = torchaudio.load(filepath)
torch.save(wav, sys.argv[2] + '.pth')

