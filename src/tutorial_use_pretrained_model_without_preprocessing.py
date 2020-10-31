# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ tutorial_use_pretrained_model_without_preprocessing.py ]
#   Synopsis     [ an example code of using the wrapper class for downstream feature extraction or finetune ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


"""
[Introduction]
This is a tutorial for using pre-trained models without doing preprocessing.
Only for pre-trained models that has `on-the-fly` in their dir name (They are trained with on-the-fly feature extractors).
"""


###############
# IMPORTATION #
###############
import torch
from transformer.nn_transformer import TRANSFORMER


################
# EXAMPLE CODE #
################
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
    'ckpt_file'     : './result/result_transformer/on-the-fly-melBase960-b12-T-libri/states-1000000.ckpt',
    'load_pretrain' : 'True',
    'no_grad'       : 'True',
    'dropout'       : 'default',
    'spec_aug'      : 'False',
    'spec_aug_prev' : 'True',
    'weighted_sum'  : 'False',
    'select_layer'  : -1,
    'permute_input' : 'False',
}
# setup the transformer model
model = TRANSFORMER(options=options, inp_dim=0) # set inp_dim to 0 for auto setup

# load raw wav
example_wav = '../LibriSpeech/test-clean/61/70970/61-70970-0000.flac'
input_wav = TRANSFORMER.load_data(example_wav, **model.config['online']) # size: (seq_len, dim) = (97200, 1)

# forward
input_wav = input_wav.unsqueeze(0) # add batch dim, size: (batch, seq_len, dim) = (1, 97200, 1)
output_repr = model(input_wav) # preprocessing of "wav -> acoustic feature" is done during forward

# show size
print('input_wav size:', input_wav.size())
print('output_repr size:', output_repr.size()) # size: (batch, seq_len, dim) = (1, 608, 768)