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
import torch


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