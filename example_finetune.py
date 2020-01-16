# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ example_finetune.py ]
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
options = {
    'ckpt_file' : 'result/result_mockingjay/mockingjay_libri_sd1337_MelBase/mockingjay-500000.ckpt',
    'load_pretrain' : True,
    'no_grad' : False,
    'dropout' : 'default'
}
model = MOCKINGJAY(options=options, inp_dim=160)

# setup your downstream class model
classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=2).cuda()

# construct the Mockingjay optimizer
params = list(model.named_parameters()) + list(classifier.named_parameters())
optimizer = get_mockingjay_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

# forward
example_inputs = torch.zeros(1200, 16, 160) # A batch of spectrograms: (time_step, batch_size, dimension)
reps = model(example_inputs) # returns: (time_step, batch_size, hidden_size)
loss = classifier(reps, torch.LongTensor([0, 1, 0]).cuda())

# update
loss.backward()
optimizer.step()

# save
PATH_TO_SAVE_YOUR_MODEL = 'example.ckpt'
states = {'Classifier': classifier.state_dict(), 'Mockingjay': model.state_dict()}
torch.save(states, PATH_TO_SAVE_YOUR_MODEL)