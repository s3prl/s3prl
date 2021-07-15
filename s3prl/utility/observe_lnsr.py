# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ observe_lnsr.py ]
#   Synopsis     [ a script for calculating the "layerwise noise to signal ratio" (LNSR) proposed in https://arxiv.org/abs/2006.03214]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import numpy as np
from pathlib import Path
from transformer.nn_transformer import TRANSFORMER


def compute_lnsr(real, adve, norm_L2=True):
    real = real.reshape(real.shape[0], -1)
    adve = adve.reshape(adve.shape[0], -1)
    l2 = np.linalg.norm(real - adve, ord=2)
    if norm_L2:
        l2 /= np.linalg.norm(real, ord=2)
    return l2


def run_over_layer(layer, real_todo, adve_todo):
    
    if layer != 'feature':
        options = {
            'ckpt_file'     : 'result/result_transformer/mockingjay/LinearLarge-libri/model-500000.ckpt',
            'load_pretrain' : 'True',
            'no_grad'       : 'True',
            'dropout'       : 'default',
            'spec_aug'      : 'False',
            'spec_aug_prev' : 'True',
            'weighted_sum'  : 'False',
            'select_layer'  : layer,
            'permute_input' : 'True',
        }
        mockingjay = TRANSFORMER(options=options, inp_dim=160)
        mockingjay.eval()

    episode = []
    for i in range(len(real_todo)):
        r = torch.FloatTensor(np.load(real_todo[i])).unsqueeze(1)
        a = torch.FloatTensor(np.load(adve_todo[i])).unsqueeze(1)
        min_len = min(r.shape[0], a.shape[0])

        if layer == 'feature':
            l2 = compute_lnsr(r[:min_len].data.cpu().numpy(), 
                              a[:min_len].data.cpu().numpy(),
                              norm_L2=True)
        else:
            r_hidd = mockingjay(r[:min_len])
            a_hidd = mockingjay(a[:min_len])
            l2 = compute_lnsr(r_hidd.data.cpu().numpy(), 
                              a_hidd.data.cpu().numpy(),
                              norm_L2=True)
        episode.append(l2)

    return np.mean(episode)


def main():
    num_layers = 12
    data_path = 'data/adversarial/' # this can be downloaded from the S3PRL google drive
    data_type = ['fgsm_8.0', 'fgsm_16.0', 'pgd_8.0', 'pgd_16.0']
    data_type = data_type[0] # select data type here
    
    real_data = os.path.join(data_path, 'original')
    adve_data = os.path.join(data_path, data_type)
    real_todo = sorted(list(Path(real_data).rglob("*.npy")))
    adve_todo = sorted(list(Path(adve_data).rglob("*.npy")))

    assert len(real_todo) == len(adve_todo)
    print('Number of data: ', len(real_todo))

    dist = run_over_layer('feature', real_todo, adve_todo)
    print('[Original v.s. ' + data_type + '] Acoustic distance: ', dist)

    dist = []
    for i in range(num_layers):
        m = run_over_layer(i, real_todo, adve_todo)
        dist.append(m)
        print('Layer: ', i+1, 'Mse: ', m)
    print('[Original v.s. ' + data_type + '] Hidden rep distance over all layers (1->12): ', dist)


if __name__ == '__main__':
    main()