import torch
import importlib


################
# EXAMPLE CODE #
################
"""
model
    Input: 
        list of unpadded wavs [wav1, wav2, ...]
        each wav is in torch.FloatTensor and on the device same as model
    Output:
        list of unpadded representations [rep1, rep2, ...]
"""
device = 'cuda'
Upstream = getattr(importlib.import_module('hubconf'), 'decoar2')
model = Upstream(refresh=True).to(device)

# example_wavs: a batch of audio in wav:  (batch_size, wav_time_step)
example_wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)]
# reps: a batch of representations: (batch_size, spectrogram_time_step, feature_size)
reps = model(example_wavs) 

print('example batch_size:', len(reps))
print('example sequence of feature:', reps[0].size())