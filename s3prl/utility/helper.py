# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utility/helper.py ]
#   Synopsis     [ helper functions ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import sys
import math
import torch
import shutil
import builtins
import numpy as np
from time import time
from typing import List
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

def is_leader_process():
    return not is_initialized() or get_rank() == 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_used_parameters(model):
    # The model should be at least backward once
    return sum(p.numel() for p in model.parameters() if p.grad is not None)

def get_time_tag():
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S')

def backup(src_path, tgt_dir):
    stem = Path(src_path).stem
    suffix = Path(src_path).suffix
    shutil.copyfile(src_path, os.path.join(tgt_dir, f'{stem}_{get_time_tag()}{suffix}'))

def get_model_state(model):
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()

def show(*args, **kwargs):
    if is_leader_process():
        print(*args, **kwargs)

def hack_isinstance():
    # Pytorch do not support passing a defaultdict into DDP module
    # https://github.com/pytorch/pytorch/blob/v1.7.1/torch/nn/parallel/scatter_gather.py#L19
    
    # This hack can be removed after torch 1.8.0, where when each DDP process use single GPU
    # (which is the best practice) DDP will not pass args, kwargs into scatter function
    # https://github.com/pytorch/pytorch/blob/v1.7.1/torch/nn/parallel/distributed.py#L617
    # https://github.com/pytorch/pytorch/blob/v1.8.0-rc1/torch/nn/parallel/distributed.py#L700

    _isinstance = builtins.isinstance
    def isinstance(obj, cls):
        if _isinstance(obj, defaultdict):
            return _isinstance(obj, cls) and issubclass(cls, defaultdict)
        return _isinstance(obj, cls)
    builtins.isinstance = isinstance

def override(string, args, config):
    """
    Example usgae:
        -o "config.optimizer.lr=1.0e-3,,config.optimizer.name='AdamW',,config.runner.eval_dataloaders=['dev', 'test']"
    """
    options = string.split(',,')
    for option in options:
        option = option.strip()
        key, value_str = option.split('=')
        key, value_str = key.strip(), value_str.strip()
        first_field, *remaining = key.split('.')

        try:
            value = eval(value_str)
        except:
            value = value_str

        print(f'[Override] - {key} = {value}', file=sys.stderr)

        if first_field == 'args':
            assert len(remaining) == 1
            setattr(args, remaining[0], value)
        elif first_field == 'config':
            target_config = config
            for i, field_name in enumerate(remaining):
                if i == len(remaining) - 1:
                    target_config[field_name] = value
                else:
                    target_config.setdefault(field_name, {})
                    target_config = target_config[field_name]

def zero_mean_unit_var_norm(input_values: List[np.ndarray]) -> List[np.ndarray]:
    """
    Every array in the list is normalized to have zero mean and unit variance
    Taken from huggingface to ensure the same behavior across s3prl and huggingface
    Reference: https://github.com/huggingface/transformers/blob/a26f4d620874b32d898a5b712006a4c856d07de1/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L81-L86
    """
    return [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in input_values]

#####################
# PARSE PRUNE HEADS #
#####################
def parse_prune_heads(config):
    if 'prune_headids' in config['transformer'] and config['transformer']['prune_headids'] != 'None':
        heads_int = []
        spans = config['transformer']['prune_headids'].split(',')
        for span in spans:
            endpoints = span.split('-')
            if len(endpoints) == 1:
                heads_int.append(int(endpoints[0]))
            elif len(endpoints) == 2:
                heads_int += torch.arange(int(endpoints[0]), int(endpoints[1])).tolist()
            else:
                raise ValueError
        print(f'[PRUNING] - heads {heads_int} will be pruned')
        config['transformer']['prune_headids'] = heads_int
    else:
        config['transformer']['prune_headids'] = None


##########################
# GET TRANSFORMER TESTER #
##########################
def get_transformer_tester(from_path='result/result_transformer/libri_sd1337_fmllrBase960-F-N-K-RA/model-1000000.ckpt', display_settings=False):
    ''' Wrapper that loads the transformer model from checkpoint path '''

    # load config and paras
    all_states = torch.load(from_path, map_location='cpu')
    config = all_states['Settings']['Config']
    paras = all_states['Settings']['Paras']
    
    # handling older checkpoints
    if not hasattr(paras, 'multi_gpu'):
        setattr(paras, 'multi_gpu', False)
    if 'prune_headids' not in config['transformer']:
        config['transformer']['prune_headids'] = None

    # display checkpoint settings
    if display_settings:
        for cluster in config:
            print(cluster + ':')
            for item in config[cluster]:
                print('\t' + str(item) + ': ', config[cluster][item])
        print('paras:')
        v_paras = vars(paras)
        for item in v_paras:
            print('\t' + str(item) + ': ', v_paras[item])

    # load model with Tester
    from transformer.solver import Tester
    tester = Tester(config, paras)
    tester.set_model(inference=True, with_head=False, from_path=from_path)
    return tester
