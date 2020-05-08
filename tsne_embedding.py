import seaborn 
import torch 
import yaml
import random
import argparse
import numpy as np
from utils.timer import Timer
import wandb

def get_mockingjay_args():
    
    parser = argparse.ArgumentParser(description='Argument Parser for the mockingjay project.')
    
    # setting
    parser.add_argument('--config', default='config/mockingjay_libri.yaml', type=str, help='Path to experiment config.')
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)

    # Logging
    parser.add_argument('--logdir', default='log/log_mockingjay_albert_3_layer_resume_5e-5/', type=str, help='Logging path.', required=False)
    parser.add_argument('--name', default=None, type=str, help='Name for logging.', required=False)

    # model ckpt[]
    parser.add_argument('--load', action='store_true', help='Load pre-trained model to restore training, no need to specify this during testing.')
    # parser.add_argument('--ckpdir', default='../result_albert/albert_2_25_mockingjay_5e-5', type=str, help='Checkpoint/Result path.', required=False)
    # parser.add_argument('--ckpdir', default='../result_albert/albert-650000/albert_3l_melbase', type=str, help='Checkpoint/Result path.', required=False)
    #parser.add_argument('--ckpdir', default='../result_albert/albert-650000/albert_6l_melbase', type=str, help='Checkpoint/Result path.', required=False)
    # parser.add_argument('--ckpdir', default='../result_albert/albert-650000/albert_3l_mask1', type=str, help='Checkpoint/Result path.', required=False)
    parser.add_argument('--ckpdir', default='../result_albert/albert-650000/ALBERT-6l-2', type=str, help='Checkpoint/Result path.', required=False)
    # parser.add_argument('--ckpdir', default='../result_albert/albert-650000/albert_12l_mask1', type=str, help='Checkpoint/Result path.', required=False)
    # parser.add_argument('--ckpdir', default='../result_albert/albert-650000/albert_6l_mask1_number1', type=str, help='Checkpoint/Result path.', required=False)
    # parser.add_argument("--ckpdir", default='../previous_result', type=str, help='Checkpoint/Result path.', required=False)
    # parser.add_argument("--ckpdir" , default='/home/pohan1996/melbase-albert/albert-3l-melbase-downsample1-consecutive1', type=str, help='Checkpoint/Result path.', required=False)
    # parser.add_argument("--ckpdir" , default='/home/pohan1996/melbase-albert/albert-3l-melbase-downsample1-consecutive20', type=str, help='Checkpoint/Result path.', required=False)
    parser.add_argument('--test_speaker_CPC', action='store_true', help='Test mel or mockingjay representations using the trained speaker classifier.')
    parser.add_argument('--test_speaker_large', action='store_true', help='Test mel or mockingjay representations using the trained speaker classifier.')
    parser.add_argument('--test_phone', action='store_true', help='Test mel or mockingjay representations using the trained speaker classifier.')
    parser.add_argument("--only_query", action="store_true", help="if true, use original mockingjay not albert")

    # parser.add_argument('--ckpt', default="mockingjay_libri_sd1337/mockingjayAlbert-490000.ckpt", type=str, help='path to mockingjay model checkpoint.', required=False)
    parser.add_argument('--ckpt', default="mockingjay_libri_sd1337/mockingjayALBERT-490000.ckpt", type=str, help='path to mockingjay model checkpoint.', required=False)
    parser.add_argument('--dckpt', default='speaker-dev-CPC/best_val.ckpt', type=str, help='path to downstream checkpoint.', required=False)

    # mockingjay
    parser.add_argument('--run_mockingjay', action='store_true', help='train and test the downstream tasks using mockingjay representations.')
    parser.add_argument('--fine_tune', action='store_true', help='fine tune the mockingjay model with downstream task.')
    
    # speaker verification task
    
    # Options
    parser.add_argument('--with_head', action='store_true', help='inference with the spectrogram head, the model outputs spectrogram.')
    parser.add_argument("--bert", action="store_true", help="if true, use original mockingjay not albert")
    parser.add_argument('--output_attention', action='store_true', help='plot attention')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')


    args = parser.parse_args()
    setattr(args,'gpu', not args.cpu)
    setattr(args,'verbose', not args.no_msg)
    config = yaml.load(open(args.config,'r'))
    config['timer'] = Timer()
    
    return config, args

def main():
    # WANDB_MODE="dryrun"
    # get arguments
    config, args = get_mockingjay_args()
    # wandb.init(config=config,project="albert-mockingjay-downstream-task")#,resume=True)
    wandb=None
    # wandb.init(config=config,project="albert-mockingjay-downstream-task",name="PHALBERT-feature-extract")#,resume=True)
    # wandb.config.update(args)
    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    from downstream.solver import Downstream_tsne_Tester
    if args.test_phone:
        task = 'mockingjay_phone' if args.run_mockingjay \
            else 'apc_phone' if args.run_apc else 'baseline_phone'
        tester = Downstream_tsne_Tester(config, args, task=task)
        tester.load_data(split='test', load='phone')
        tester.set_model(inference=True)
        tester.exec()
    elif args.test_speaker_CPC:
        task = 'mockingjay_speakerCPC' if args.run_mockingjay \
                else 'apc_speaker' if args.run_apc else 'baseline_speaker'
        tester = Downstream_tsne_Tester(config, args, task=task)
        tester.load_data(split='test', load='speakerCPC')
        tester.set_model(inference=True)
        tester.exec()
    elif args.test_speaker_large:
        
        from downstream.solver import Downstream_tsne_Tester
        task = 'mockingjay_speakerlarge' if args.run_mockingjay \
                else 'apc_speakerlarge' if args.run_apc else 'baseline_speakerlarge'
        tester = Downstream_tsne_Tester(config, args, task=task)
        tester.load_data(split='test', load='speakerlarge')
        tester.set_model(inference=True)
        tester.exec()
if __name__ == "__main__":
    main()
