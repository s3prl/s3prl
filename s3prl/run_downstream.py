import os
import yaml
import glob
import torch
import random
import logging
import argparse
import torchaudio
import numpy as np
from pathlib import Path
from argparse import Namespace
from torch.distributed import is_initialized, get_world_size, get_rank

from s3prl import hub
from s3prl.downstream.runner import Runner
from s3prl.utility.helper import backup, get_unique_tag, hack_isinstance, override

from huggingface_hub import HfApi, HfFolder
log = logging.getLogger(__name__)
CANNOT_LOAD_FROM_CKPT_ARGS = [
    "mode", "evaluate_split", "override",
    "backend", "past_exp", "verbose", "expdir", "expname",
]

def get_downstream_args():
    parser = argparse.ArgumentParser()

    # train or test for this experiment
    parser.add_argument('-m', '--mode', choices=['train', 'evaluate', 'inference'], required=True)
    parser.add_argument('-t', '--evaluate_split', default='test')
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')

    # distributed training
    parser.add_argument('--backend', default='nccl', help='The backend for distributed training')

    # use a ckpt as the experiment initialization
    # if set, all the args and config below this line will be overwrited by the ckpt
    # if a directory is specified, the latest ckpt will be used by default
    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')

    # only load the parameters in the checkpoint without overwriting arguments and config, this is for evaluation
    parser.add_argument('-i', '--init_ckpt', metavar='CKPT_PATH', help='Load the checkpoint for evaluation')

    # configuration for the experiment, including runner and downstream
    parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment except the upstream model')

    # downstream settings
    parser.add_argument('-d', '--downstream', help='\
        Typically downstream dataset need manual preparation.\
        Please check downstream/README.md for details'
    )
    parser.add_argument('-v', '--downstream_variant', help='Downstream vairants given the same expert')

    # upstream settings
    parser.add_argument('--hub', default="torch", choices=["torch", "huggingface"],
        help='The model Hub used to retrieve the upstream model.')

    upstreams = [attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream',  help=""
        'Upstreams with \"_local\" or \"_url\" postfix need local ckpt (-k) or config file (-g). '
        'Other upstreams download two files on-the-fly and cache them, so just -u is enough and -k/-g are not needed. '
        'Please check upstream/README.md for details. '
        f"Available options in S3PRL: {upstreams}. "
    )
    parser.add_argument('-k', '--upstream_ckpt', metavar='{PATH,URL,GOOGLE_DRIVE_ID}', help='Only set when the specified upstream need it')
    parser.add_argument('-g', '--upstream_model_config', help='The config file for constructing the pretrained model')
    parser.add_argument('-r', '--upstream_refresh', action='store_true', help='Re-download cached ckpts for on-the-fly upstream variants')
    parser.add_argument('-f', '--upstream_trainable', action='store_true', help='Fine-tune, set upstream.train(). Default is upstream.eval()')
    parser.add_argument('-s', '--upstream_feature_selection', default='hidden_states', help='Specify the layer to be extracted as the representation')
    parser.add_argument('-l', '--upstream_layer_selection', type=int, help='Select a specific layer for the features selected by -s')
    parser.add_argument('--upstream_feature_normalize', action='store_true', help='Specify whether to normalize hidden features before weighted sum')
    parser.add_argument('--upstream_model_name', default="model.pt", help='The name of the model file in the HuggingFace Hub repo.')
    parser.add_argument('--upstream_revision', help="The commit hash of the specified HuggingFace Repository")

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/downstream
    parser.add_argument('-n', '--expname', help='Save experiment at result/downstream/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')
    parser.add_argument('-a', '--auto_resume', action='store_true', help='Auto-resume if the expdir contains checkpoints')
    parser.add_argument('--push_to_hf_hub', default=False, help='Push all files in experiment directory to the Hugging Face Hub. To use this feature you must set HF_USERNAME and HF_PASSWORD as environment variables in your shell')
    parser.add_argument('--hf_hub_org', help='The Hugging Face Hub organisation to push fine-tuned models to')

    # options
    parser.add_argument('--verbose', type=int, default=1, help='logging level: 1 INFO, 2 DEBUG, others WARNING')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--cache_dir', help='The cache directory for pretrained model downloading')
    parser.add_argument('--disable_cudnn', action='store_true', help='Disable CUDNN')
    parser.add_argument('--sharing_strategy', default="file_system")
    parser.add_argument('--audio_backend', default="sox_io")
    parser.add_argument('--local_rank', type=int)

    args = parser.parse_args()
    backup_files = []
    messages = []

    if args.expdir is None:
        args.expdir = f'result/downstream/{args.expname}'
    else:
        args.expname = Path(args.expdir).stem

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
        messages.append(f'Resume from {ckpt_pth}')
        args.expdir = Path(ckpt_pth).parent
        args.expname = Path(args.expdir).parts[-1]

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
        args = update_args(args, ckpt['Args'], preserve_list=CANNOT_LOAD_FROM_CKPT_ARGS)
        os.makedirs(args.expdir, exist_ok=True)
        args.init_ckpt = ckpt_pth
        config = ckpt['Config']

    else:
        messages.append('Start a new experiment')
        os.makedirs(args.expdir, exist_ok=True)

        if args.config is None:
            args.config = f'./downstream/{args.downstream}/config.yaml'
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        if args.upstream_model_config is not None and os.path.isfile(args.upstream_model_config):
            backup_files.append(args.upstream_model_config)

    assert "None" not in str(args.expdir) and "None" not in str(args.expname), (
        "When launching a new experiment, -p or -n must be specified. "
        "When resuming from a existing experiment directory, -p must be specified. "
        "When resuming from a trained checkpoint, -e must be specified."
    )
    messages.append(f'Set expdir as {args.expdir}, expname as {args.expname}')
    messages.append(f'Results can be found at: {args.expdir}')

    if args.override is not None and args.override.lower() != "none":
        override_msgs = override(args.override, args, config)
        for override_msg in override_msgs:
            messages.append(override_msg)
        os.makedirs(args.expdir, exist_ok=True)

    return args, config, backup_files, messages


def main():
    logging.basicConfig(level=logging.INFO)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torchaudio.set_audio_backend('sox_io')
    hack_isinstance()

    # get config and arguments
    args, config, backup_files, messages = get_downstream_args()
    torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)
    torchaudio.set_audio_backend(args.audio_backend)

    # setup DistributedDataParallel
    if args.local_rank is not None:
        # When torch.distributed.launch is used
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    local_rank = os.environ.get("LOCAL_RANK")
    if isinstance(local_rank, str):
        print(f"Environment variable LOCAL_RANK: {local_rank} detected. "
               "Use DistributedDataParallel to train the model")
        assert local_rank.isnumeric(), local_rank

        local_rank = int(local_rank)
        device_count = torch.cuda.device_count()
        assert local_rank < device_count, f"{local_rank} !< {device_count}"

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(args.backend)

    if args.mode == 'train' and args.past_exp:
        ckpt = torch.load(args.init_ckpt, map_location='cpu')

        now_use_ddp = is_initialized()
        original_use_ddp = ckpt.get("WorldSize") is not None
        assert now_use_ddp == original_use_ddp, f'{now_use_ddp} != {original_use_ddp}'

        if now_use_ddp:
            now_world = get_world_size()
            original_world = ckpt['WorldSize']
            assert now_world == original_world, f'{now_world} != {original_world}'

    # logging level
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose == 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    # logging format
    root_log = logging.getLogger()
    root_log.setLevel(level)
    rank_string = f"RANK {get_rank()} " if is_initialized() else ""
    formatter = logging.Formatter(f"[%(levelname)s] {rank_string}%(asctime)s (%(module)s.%(funcName)s:%(lineno)d): %(message)s")

    # logging file
    log_file = Path(args.expdir) / f"log_{get_unique_tag()}"
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(formatter)
    root_log.addHandler(fileHandler)
    messages.append(f"The log file can be found at: {log_file}")

    # logging stream
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    root_log.addHandler(streamHandler)

    if level == logging.WARNING:
        log.warning("Skip DEBUG/INFO messages in S3PRL")

    for message in messages:
        log.info(message)

    # save command
    args_file = f"args_{get_unique_tag()}.yaml"
    log.info(f"The args can be found at: {args_file}")
    with (Path(args.expdir) / args_file).open("w") as file:
        yaml.dump(vars(args), file)

    config_file = f"config_{get_unique_tag()}.yaml"
    log.info(f"The config can be found at: {config_file}")
    with (Path(args.expdir) / config_file).open("w") as file:
        yaml.dump(config, file)

    for file in backup_files:
        backup(file, args.expdir)

    if args.cache_dir is not None:
        torch.hub.set_dir(args.cache_dir)

    if args.hub == "huggingface":
        args.from_hf_hub = True
        # Setup auth
        hf_user = os.environ.get("HF_USERNAME")
        hf_password = os.environ.get("HF_PASSWORD")
        huggingface_token = HfApi().login(username=hf_user, password=hf_password)
        HfFolder.save_token(huggingface_token)
        log.info(f"Logged into Hugging Face Hub with user: {hf_user}")

    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    runner = Runner(args, config)
    eval(f'runner.{args.mode}')()


if __name__ == '__main__':
    main()
