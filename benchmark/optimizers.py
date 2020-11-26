import torch
import torch.nn as nn

from torch.optim import Adam
from transformer.optimization import BertAdam, Lamb


def get_grouped_parameters(optimized_models):
    named_params = []
    for m in optimized_models:
        named_params += list(m.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    return grouped_parameters


def get_BertAdam(optimized_models, lr=2e-4, total_steps=20000, warmup_proportion=0.07, **kwargs):
    grouped_parameters = get_grouped_parameters(optimized_models)
    optimizer = BertAdam(grouped_parameters, lr=lr,
                         warmup=warmup_proportion,
                         t_total=total_steps)
    return optimizer


def get_AdamW(optimized_models, lr=2e-4, total_steps=20000, warmup_proportion=0.07, **kwargs):
    grouped_parameters = get_grouped_parameters(optimized_models)
    optimizer = Lamb(grouped_parameters,
                     lr=lr,
                     warmup=warmup_proportion,
                     t_total=total_steps,
                     adam=True,
                     correct_bias=True)
    return optimizer


def get_Lamb(optimized_models, lr=2e-4, total_steps=20000, warmup_proportion=0.07, **kwargs):
    grouped_parameters = get_grouped_parameters(optimized_models)
    optimizer = Lamb(grouped_parameters,
                     lr=lr,
                     warmup=warmup_proportion,
                     t_total=total_steps,
                     adam=False,
                     correct_bias=False)
    return optimizer


def get_Adam(optimized_models, lr=2e-4, **kwargs):
    params = []
    for m in optimized_models:
        params += list(m.parameters())
    return Adam(params, lr=lr, betas=(0.9, 0.999))
