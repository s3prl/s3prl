# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ example_extract_finetune.py ]
#   Synopsis     [ an example code of using the wrapper class for downstream feature extraction or finetune ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from mockingjay.nn_mockingjay import MOCKINGJAY
from downstream.model import example_classifier
from downstream.solver import get_mockingjay_optimizer

################
# EXAMPLE CODE #
################

# setup the mockingjay model
"""
`options`: a python dictionary containing the following keys:
    ckpt_file: str, a path specifying the pre-trained ckpt file
    load_pretrain: str, ['True', 'False'], whether to load pre-trained weights
    no_grad: str, ['True', 'False'], whether to have gradient flow over this class
    dropout: float/str, use float to modify dropout value during downstream finetune, or use the str `default` for pre-train default values
    spec_aug: str, ['True', 'False'], whether to apply SpecAugment on inputs (used for ASR training)
    spec_aug_prev: str, ['True', 'False'], apply spec augment on input acoustic features if True, else apply on output representations (used for ASR training)
    weighted_sum: str, ['True', 'False'], whether to use a learnable weighted sum to integrate hidden representations from all layers, if False then use the last
    select_layer: int, select from all hidden representations, set to -1 to select the last (will only be used when weighted_sum is False)
"""
options = {
    'ckpt_file'     : './result/result_mockingjay/libri_sd1337_fmllrBase960-F-N-K-RA/model-1000000.ckpt',
    'load_pretrain' : 'True',
    'no_grad'       : 'True',
    'dropout'       : 'default',
    'spec_aug'      : 'False',
    'spec_aug_prev' : 'True',
    'weighted_sum'  : 'False',
    'select_layer'  : -1,
}
mockingjay = MOCKINGJAY(options=options, inp_dim=160)

# setup your downstream class model
classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=2).cuda()

# construct the Mockingjay optimizer
params = list(mockingjay.named_parameters()) + list(classifier.named_parameters())
optimizer = get_mockingjay_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

# forward
example_inputs = torch.zeros(1200, 3, 160) # A batch of spectrograms: (time_step, batch_size, dimension)
reps = mockingjay(example_inputs) # returns: (time_step, batch_size, hidden_size)
reps = reps.permute(1, 0, 2) # change to: (batch_size, time_step, feature_size)
labels = torch.LongTensor([0, 1, 0]).cuda()
loss = classifier(reps, labels)

# update
loss.backward()
optimizer.step()

# save
PATH_TO_SAVE_YOUR_MODEL = 'example.ckpt'
states = {'Classifier': classifier.state_dict(), 'Mockingjay': mockingjay.model.state_dict()}
torch.save(states, PATH_TO_SAVE_YOUR_MODEL)