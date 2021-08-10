# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ run_pretrain.py ]
#   Synopsis     [ scripts for running the pre-training of upstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import re
import yaml
import glob
import random
import argparse
import importlib
from shutil import copyfile
from argparse import Namespace
#-------------#
import torch
import numpy as np
#-------------#
from pretrain.runner import Runner


######################
# PRETRAIN ARGUMENTS #
######################
def get_pretrain_args():
    parser = argparse.ArgumentParser()

    # use a ckpt as the experiment initialization
    # if set, all the following args and config will be overwrited by the ckpt, except args.mode
    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')

    # configuration for the experiment, including runner and downstream
    parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment, except the upstream model')

    # upstream settings
    parser.add_argument('-u', '--upstream', choices=os.listdir('pretrain/'))
    parser.add_argument('-g', '--upstream_config', default='', metavar='PATH', help='Only set when the specified upstream need it')

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/pretrain
    parser.add_argument('-n', '--expname', help='Save experiment at result/pretrain/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--multi_gpu', action='store_true', help='Enables multi-GPU training')

    args = parser.parse_args()

    if args.past_exp:
        # determine checkpoint path
        if os.path.isdir(args.past_exp):
            ckpt_pths = glob.glob(f'{args.past_exp}/states-*.ckpt')
            assert len(ckpt_pths) > 0
            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            ckpt_pth = ckpt_pths[-1]
        else:
            ckpt_pth = args.past_exp
        
        # load checkpoint
        ckpt = torch.load(ckpt_pth, map_location='cpu')

        def update_args(old, new):
            old_dict = vars(old)
            new_dict = vars(new)
            old_dict.update(new_dict)
            return Namespace(**old_dict)

        # overwrite args and config
        args = update_args(args, ckpt['Args'])
        config = ckpt['Runner']
        args.past_exp = ckpt_pth

    else:
        if args.expdir is None:
            args.expdir = f'result/pretrain/{args.expname}'
        os.makedirs(args.expdir, exist_ok=True)

        upstream_dirs = [u for u in os.listdir('pretrain/') if re.search(f'^{u}_|^{u}$', args.upstream)]
        assert len(upstream_dirs) == 1

        if args.config is None:
            args.config = f'pretrain/{upstream_dirs[0]}/config_runner.yaml'
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        copyfile(args.config, f'{args.expdir}/config_runner.yaml')
        
        default_upstream_config = f'pretrain/{upstream_dirs[0]}/config_model.yaml'
        if args.upstream_config == '' and os.path.isfile(default_upstream_config):
            args.upstream_config = default_upstream_config
        if os.path.isfile(args.upstream_config):
            copyfile(args.upstream_config, f'{args.expdir}/config_model.yaml')

    return args, config


########
# MAIN #
########
def main():
    # get config and arguments
    args, config = get_pretrain_args()

    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    runner = Runner(args, config)
    eval('runner.train')()
    runner.logger.close()


if __name__ == '__main__':
    main()