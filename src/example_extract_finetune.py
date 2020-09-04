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
from transformer.nn_transformer import TRANSFORMER
from downstream.model import example_classifier
from downstream.solver import get_optimizer

################
# EXAMPLE CODE #
################

# setup the transformer model
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
    permute_input: str, ['True', 'False'], this attribute is for the forward method. If Ture then input ouput is in the shape of (T, B, D), if False then in (B, T, D)
"""
options = {
    'ckpt_file'     : './result/result_transformer/tera/fmllrBase960-F-N-K-libri/states-1000000.ckpt',
    'load_pretrain' : 'True',
    'no_grad'       : 'True',
    'dropout'       : 'default',
    'spec_aug'      : 'False',
    'spec_aug_prev' : 'True',
    'weighted_sum'  : 'False',
    'select_layer'  : -1,
    'permute_input' : 'False',
}
transformer = TRANSFORMER(options=options, inp_dim=40)

# setup your downstream class model
classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=2).cuda()

# construct the optimizer
params = list(transformer.named_parameters()) + list(classifier.named_parameters())
optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

# forward
example_inputs = torch.zeros(3, 1200, 40) # A batch of spectrograms:  (batch_size, time_step, feature_size)
reps = transformer(example_inputs) # returns: (batch_size, time_step, feature_size)
labels = torch.LongTensor([0, 1, 0]).cuda()
loss = classifier(reps, labels)

# update
loss.backward()
optimizer.step()

# save
PATH_TO_SAVE_YOUR_MODEL = 'example.ckpt'
states = {'Classifier': classifier.state_dict(), 'Transformer': transformer.state_dict()}
# torch.save(states, PATH_TO_SAVE_YOUR_MODEL)