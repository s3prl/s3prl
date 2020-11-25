import torch
import torch.nn as nn

from torch.optim import Adam
from transformer.optimization import BertAdam


def get_optimizer(optimized_models, optimizer_type='Adam', total_steps=20000, lr=2e-4, warmup_proportion=0.07, **kwargs):
    params, named_params = [], []
    for m in optimized_models:
        params += list(m.parameters())
        named_params += list(m.named_parameters())

    if optimizer_type == 'BertAdam':
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(grouped_parameters, lr=lr,
                             warmup=warmup_proportion,
                             t_total=total_steps)

    elif optimizer_type == 'Adam':
        optimizer = Adam(params, lr=lr, betas=(0.9, 0.999))

    return optimizer
