import torch
import torch.nn as nn

from torch.optim import Adam
from transformer.optimization import BertAdam


def get_BertAdam(optimized_models, lr=2e-4, total_steps=20000, warmup_proportion=0.07, **kwargs):
    named_params = []
    for m in optimized_models:
        named_params += list(m.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = BertAdam(grouped_parameters, lr=lr,
                         warmup=warmup_proportion,
                         t_total=total_steps)
    return optimizer


def get_Adam(optimized_models, lr=2e-4, **kwargs):
    params = []
    for m in optimized_models:
        params += list(m.parameters())
    return Adam(params, lr=lr, betas=(0.9, 0.999))
