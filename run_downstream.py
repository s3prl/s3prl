import os
import re
import sys
import yaml
import glob
import torch
import random
import argparse
import importlib
import torchaudio
import numpy as np
from argparse import Namespace
from torch.distributed import is_initialized, get_world_size

import hubconf
from downstream.runner import Runner
from utility.helper import backup, get_time_tag, hack_isinstance, is_leader_process, override


def get_downstream_args():
    parser = argparse.ArgumentParser()

    # train or test for this experiment
    parser.add_argument('-m', '--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('-t', '--evaluate_split', default='test')
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')

    # distributed training
    parser.add_argument('--backend', default='nccl', help='The backend for distributed training')
    parser.add_argument('--local_rank', type=int,
                        help=f'The GPU id this process should use while distributed training. \
                               None when not launched by torch.distributed.launch')

    # use a ckpt as the experiment initialization
    # if set, all the args and config below this line will be overwrited by the ckpt
    # if a directory is specified, the latest ckpt will be used by default
    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')

    # only load the parameters in the checkpoint without overwriting arguments and config, this is for evaluation
    parser.add_argument('-i', '--init_ckpt', metavar='CKPT_PATH', help='Load the checkpoint for evaluation')

    # configuration for the experiment, including runner and downstream
    parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment except the upstream model')

    # downstream settings
    downstreams = [item for item in os.listdir('./downstream') if os.path.isfile(os.path.join('./downstream', item, 'expert.py'))]
    parser.add_argument('-d', '--downstream', choices=downstreams, help='\
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
    parser.add_argument('-g', '--upstream_model_config', help='The config file for constructing the pretrained model')
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
    parser.add_argument('--cache_dir', help='The cache directory for pretrained model downloading')
    parser.add_argument('--verbose', action='store_true', help='Print model infomation')

    args = parser.parse_args()
    backup_files = []

    if args.expdir is None:
        args.expdir = f'result/downstream/{args.expname}'

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

        def update_args(old, new, preserve_list=None):
            out_dict = vars(old)
            new_dict = vars(new)
            for key in list(new_dict.keys()):
                if key in preserve_list:
                    new_dict.pop(key)
            out_dict.update(new_dict)
            return Namespace(**out_dict)

        # overwrite args
        cannot_overwrite_args = [
            'mode', 'evaluate_split', 'override',
            'backend', 'local_rank', 'past_exp',
        ]
        args = update_args(args, ckpt['Args'], preserve_list=cannot_overwrite_args)
        os.makedirs(args.expdir, exist_ok=True)
        args.init_ckpt = ckpt_pth
        config = ckpt['Config']

    else:
        print('[Runner] - Start a new experiment')
        os.makedirs(args.expdir, exist_ok=True)

        if args.config is None:
            args.config = f'./downstream/{args.downstream}/config.yaml'
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        if args.upstream_model_config is not None and os.path.isfile(args.upstream_model_config):
            backup_files.append(args.upstream_model_config)

    if args.override:
        override(args.override, args, config)
    
    return args, config, backup_files


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torchaudio.set_audio_backend('sox_io')
    hack_isinstance()

    # get config and arguments
    args, config, backup_files = get_downstream_args()
    if args.cache_dir is not None:
        torch.hub.set_dir(args.cache_dir)

    # When torch.distributed.launch is used
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(args.backend)

    if args.mode == 'train' and args.past_exp:
        ckpt = torch.load(args.init_ckpt, map_location='cpu')

        now_use_ddp = is_initialized()
        original_use_ddp = ckpt['Args'].local_rank is not None
        assert now_use_ddp == original_use_ddp, f'{now_use_ddp} != {original_use_ddp}'

        if now_use_ddp:
            now_world = get_world_size()
            original_world = ckpt['WorldSize']
            assert now_world == original_world, f'{now_world} != {original_world}'
    
    # Save command
    if is_leader_process():
        with open(os.path.join(args.expdir, f'args_{get_time_tag()}.yaml'), 'w') as file:
            yaml.dump(vars(args), file)

        with open(os.path.join(args.expdir, f'config_{get_time_tag()}.yaml'), 'w') as file:
            yaml.dump(config, file)

        for file in backup_files:
            backup(file, args.expdir)

    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    runner = Runner(args, config)
    eval(f'runner.{args.mode}')()


if __name__ == '__main__':
    main()
