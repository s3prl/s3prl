import os
import re
import yaml
import glob
import torch
import random
import argparse
import importlib
import numpy as np
from shutil import copyfile
from argparse import Namespace

import hubconf
from downstream.runner import Runner


def get_downstream_args():
    parser = argparse.ArgumentParser()

    # train or test for this experiment
    parser.add_argument('-m', '--mode', choices=['train', 'evaluate'], required=True)

    # use a ckpt as the experiment initialization
    # if set, all the following args and config will be overwrited by the ckpt, except args.mode
    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')

    # only load the parameters in the checkpoint without overwriting arguments and config, this is for evaluation
    parser.add_argument('-i', '--init_ckpt', metavar='CKPT_PATH', help='Load the checkpoint for evaluation')

    # configuration for the experiment, including runner and downstream
    parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment except the upstream model')

    # downstream settings
    parser.add_argument('-d', '--downstream', choices=os.listdir('./downstream'), help='\
        Typically downstream dataset need manual preparation.\
        Please check downstream/README.md for details'
    )
    parser.add_argument('-v', '--downstream_variant', help='Downstream vairants given the same expert')

    # upstream settings
    upstreams = [attr for attr in dir(hubconf) if callable(getattr(hubconf, attr)) and attr[0] != '_']
    parser.add_argument('-u', '--upstream', choices=upstreams, help='\
        Some upstream variants need local ckpt or config file.\
        Some download needed files on-the-fly and cache them.\
        Please check downstream/README.md for details'
    )
    parser.add_argument('-s', '--upstream_feature_selection', help='Specify the layer to be extracted as the representation')
    parser.add_argument('-r', '--upstream_refresh', action='store_true', help='Re-download cached ckpts for on-the-fly upstream variants')
    parser.add_argument('-k', '--upstream_ckpt', metavar='{PATH,URL,GOOGLE_DRIVE_ID}', help='Only set when the specified upstream need it')
    parser.add_argument('-f', '--upstream_trainable', action='store_true', help='Fine-tune, set upstream.train(). Default is upstream.eval()')

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/downstream
    parser.add_argument('-n', '--expname', help='Save experiment at result/downstream/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')
    parser.add_argument('-a', '--auto_resume', action='store_true', help='Auto-resume if the expdir contains checkpoints')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')

    args = parser.parse_args()

    if args.expdir is None:
        args.expdir = f'result/downstream/{args.expname}'

    if os.path.isfile(f'{args.expdir}/{args.mode}_finished'):
        exit(0)

    if args.auto_resume:
        if os.path.isdir(args.expdir):
            ckpt_pths = glob.glob(f'{args.expdir}/states-*.ckpt')
            if len(ckpt_pths) > 0:
                args.past_exp = args.expdir

    if args.past_exp:
        # determine checkpoint path
        if os.path.isdir(args.past_exp):
            ckpt_pths = glob.glob(f'{args.past_exp}/states-*.ckpt')
            assert len(ckpt_pths) > 0
            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            ckpt_pth = ckpt_pths[-1]
        else:
            ckpt_pth = args.past_exp

        print(f'[Runner] - Resume from {ckpt_pth}')

        # load checkpoint
        ckpt = torch.load(ckpt_pth, map_location='cpu')

        def update_args(old, new):
            old_dict = vars(old)
            new_dict = vars(new)
            old_dict.update(new_dict)
            return Namespace(**old_dict)

        # overwrite args and config
        mode = args.mode
        args = update_args(args, ckpt['Args'])
        config = ckpt['Config']
        args.mode = mode
        args.init_ckpt = ckpt_pth

    else:
        print('[Runner] - Start a new experiment')
        os.makedirs(args.expdir, exist_ok=True)

        if args.config is None:
            args.config = f'./downstream/{args.downstream}/config.yaml'
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        copyfile(args.config, f'{args.expdir}/config.yaml')

    return args, config


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

    runner = Runner(args, config)
    eval(f'runner.{args.mode}')()
    runner.logger.close()


if __name__ == '__main__':
    main()
