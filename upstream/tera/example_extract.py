# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/tera/example_extract.py ]
#   Synopsis     [ an example code of using the wrapper class for downstream feature extraction ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
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


# below is an example of using the online available checkpoint
Upstream = getattr(importlib.import_module('hubconf'), 'tera')
model = Upstream(refresh=True).to(device)


# below is an example of using a local checkpoint
try:
    path = './s3prl/tera/logMelBase-T-F-M-AdamW-b32-1m-960hr-drop1/states-1000000.ckpt'
    Upstream_local = getattr(importlib.import_module('hubconf'), 'tera_local')
    model_local = Upstream_local(ckpt=path).to(device)
except:
    print(f'The path {path} is not a valid checkpoint!')


# example_wavs: a batch of audio in wav:  (batch_size, wav_time_step)
example_wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)]
# reps: a batch of representations: (batch_size, spectrogram_time_step, feature_size)
reps = model(example_wavs)

print('example batch_size:', len(reps))
print('example sequence of feature:', reps[0].size())