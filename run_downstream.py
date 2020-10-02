# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ run_downstream.py ]
#   Synopsis     [ scripts for running downstream evaluation of upstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import yaml
import torch
import random
import argparse
import numpy as np
from shutil import copyfile
from dataloader import get_Dataloader
from transformer.nn_transformer import TRANSFORMER, DUAL_TRANSFORMER
from downstream.model import dummy_upstream, LinearClassifier, RnnClassifier
from downstream.runner import Runner


"""
USAGE:
    You can easily initialize your upstream pre-trained model in the function `get_upstream_model()`.
    There are only three simple requirements for each upstream model:
        1) Implement the `forward` method of `nn.Module`,
        2) Contains the `out_dim` attribute.
        3) Takes input and output in the shape of: (batch_size, time_steps, feature_dim)
    then you can use these scipts to evaluate your model.
"""


#################
# PATH HANDLING #
#################
import sys
S3PRL_PATH = os.getcwd() # or set this to your own path that points to the S3PRL repo
if S3PRL_PATH not in sys.path:
    sys.path.append(S3PRL_PATH)


########################
# DOWNSTREAM ARGUMENTS #
########################
def get_downstream_args():
    
    parser = argparse.ArgumentParser(description='Argument Parser for Downstream Tasks of the S3PLR project.')
    
    # required
    parser.add_argument('--run',  choices=['phone_linear', 'phone_1hidden', 'phone_concat', 'speaker_frame', 'speaker_utterance'], help='select task.', required=True)

    # upstream settings
    parser.add_argument('--ckpt', default='', type=str, help='Path to upstream pre-trained checkpoint, required if using other than baseline', required=False)
    parser.add_argument('--upstream', choices=['dual_transformer', 'transformer', 'apc', 'baseline'], default='baseline', help='Whether to use upstream models for speech representation or fine-tune.', required=False)
    parser.add_argument('--input_dim', default=0, type=int, help='Input dimension used to initialize transformer models', required=False)
    parser.add_argument('--fine_tune', action='store_true', help='Whether to fine tune the transformer model with downstream task.', required=False)
    parser.add_argument('--weighted_sum', action='store_true', help='Whether to use weighted sum on the transformer model with downstream task.', required=False)
    parser.add_argument('--dual_mode',choices=['phone', 'speaker', 'phone speaker'], default='phone', help='Whether to use weighted sum on the transformer model with downstream task.', required=False)
    parser.add_argument('--online_config', default=None, help='Explicitly specify the config of on-the-fly feature extraction')

    # Options
    parser.add_argument('--name', default=None, type=str, help='Name of current experiment.', required=False)
    parser.add_argument('--config', default='config/downstream.yaml', type=str, help='Path to downstream experiment config.', required=False)
    parser.add_argument('--phone_set', choices=['cpc_phone', 'montreal_phone'], default='cpc_phone', help='Phone set for phone classification tasks', required=False)
    parser.add_argument('--expdir', default='', type=str, help='Path to store experiment result, if empty then default is used.', required=False)
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')

    # parse
    args = parser.parse_args()
    setattr(args, 'gpu', not args.cpu)
    setattr(args, 'task', args.phone_set if 'phone' in args.run else 'speaker')
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if args.online_config is not None:
        online_config = yaml.load(open(args.online_config, 'r'), Loader=yaml.FullLoader)
        args.online_config = online_config
    
    return args, config


################
# GET UPSTREAM #
################
"""
Upstream model should meet the requirement of:
    1) Implement the `forward` method of `nn.Module`,
    2) Contains the `out_dim` attribute.
    3) Takes input and output in the shape of: (batch_size, time_steps, feature_dim)
"""
def get_upstream_model(args):
    
    print('[run_downstream] - getting upstream model:', args.upstream)

    if args.upstream == 'transformer' or args.upstream == 'dual_transformer':
        options = {'ckpt_file'     : args.ckpt,
                   'load_pretrain' : 'True',
                   'no_grad'       : 'True' if not args.fine_tune else 'False',
                   'dropout'       : 'default',
                   'spec_aug'      : 'False',
                   'spec_aug_prev' : 'True',
                   'weighted_sum'  : 'True' if args.weighted_sum else 'False',
                   'select_layer'  : -1,
                   'permute_input' : 'False'
        }

    if args.upstream == 'transformer':
        upstream_model = TRANSFORMER(options, args.input_dim, online_config=args.online_config)
    
    elif args.upstream == 'dual_transformer':
        upstream_model = DUAL_TRANSFORMER(options, args.input_dim, mode=args.dual_mode)
        
    elif args.upstream == 'apc':
        raise NotImplementedError

    elif args.upstream == 'baseline':
        upstream_model = dummy_upstream(args.input_dim)

    else:
        raise NotImplementedError ######### plug in your upstream pre-trained model here #########

    assert(hasattr(upstream_model, 'forward'))
    assert(hasattr(upstream_model, 'out_dim'))
    return upstream_model


##################
# GET DATALOADER #
##################
def get_dataloader(args, dataloader_config):
    pretrain_config = torch.load(args.ckpt, map_location='cpu')['Settings']['Config']
    if 'online' in pretrain_config:
        dataloader_config['online'] = pretrain_config['online']
    elif args.online_config is not None:
        dataloader_config['online'] = args.online_config

    if not os.path.exists(dataloader_config['data_path']):
        raise RuntimeError('[run_downstream] - Data path not valid:', dataloader_config['data_path'])    
    print('[run_downstream] - Loading input data: ' + str(dataloader_config['train_set']) + ' from ' + dataloader_config['data_path'])
    
    if args.task == 'speaker':
        print('[run_downstream] - Loading speaker data: ' + str(dataloader_config['train_set']) + ' from ' + dataloader_config['data_path'])
    else:
        print('[run_downstream] - Loading phone data: ' + dataloader_config['phone_path'])
        if not os.path.exists(dataloader_config['phone_path']):
            raise RuntimeError('[run_downstream] - Phone path not valid:', dataloader_config['phone_path'])
        if args.task == 'montreal_phone':
            print('[run_downstream] - WARNING: Using a non-preset phone set! Please make sure \'data_path\' (should be: data/libri_mel160_subword5000) and \'phone_path\' (should be: data/libri_phone) are set correctly.')

    print('[run_downstream] - getting train dataloader...')
    train_loader = get_Dataloader(split='train', load=args.task, use_gpu=args.gpu, seed=args.seed, **dataloader_config)

    print('[run_downstream] - getting dev dataloader...')
    dev_loader = get_Dataloader(split='dev', load=args.task, use_gpu=args.gpu, seed=args.seed, **dataloader_config)

    print('[run_downstream] - getting test dataloader...')
    test_loader = get_Dataloader(split='test', load=args.task, use_gpu=args.gpu, seed=args.seed, **dataloader_config)
    
    return train_loader, dev_loader, test_loader


##################
# GET DOWNSTREAM #
##################
def get_downstream_model(args, input_dim, class_num, config):
    
    model_name = args.run.split('_')[-1].replace('frame', 'linear') # support names: ['linear', '1hidden', 'concat', 'utterance']
    model_config = config['model'][model_name]

    if args.task == 'speaker' and 'utterance' in args.run:
        downstream_model = RnnClassifier(input_dim, class_num, model_config)
    else:
        downstream_model = LinearClassifier(input_dim, class_num, model_config)
    
    return downstream_model


########
# MAIN #
########
def main():
    
    # get config and arguments
    args, config = get_downstream_args()
    
    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # mkdir
    if args.expdir == '':
        if args.name is None: args.name = 'exp_' + str(random.randint(0, 999))
        expdir = os.path.join('result/result_' + args.upstream + '_' + args.task + '/', args.name)
    else:
        expdir = args.expdir
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    copyfile(args.config, os.path.join(expdir, args.config.split('/')[-1]))

    # get upstream model
    upstream_model = get_upstream_model(args) ######### plug in your upstream pre-trained model here #########

    # get dataloaders
    train_loader, dev_loader, test_loader = get_dataloader(args, config['dataloader'])

    # get downstream model
    downstream_model = get_downstream_model(args, upstream_model.out_dim, train_loader.dataset.class_num, config)

    # train
    runner = Runner(args=args,
                    runner_config=config['runner'],
                    dataloader= {'train':train_loader, 'dev':dev_loader, 'test':test_loader}, 
                    upstream=upstream_model, 
                    downstream=downstream_model, 
                    expdir=expdir)
    runner.set_model()
    runner.train()


if __name__ == '__main__':
    main()