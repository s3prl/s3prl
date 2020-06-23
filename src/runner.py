# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ src/runner.py ]
#   Synopsis     [ scripts for running pre-training and downstream evaluation of transformer models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


"""
WARNING:
    This script is deprecated,
    we suggest you use the new scripts of: `run_upstream.py` and `run_downstream.py`
"""


###############
# IMPORTATION #
###############
import yaml
import torch
import random
import argparse
import numpy as np
from utility.helper import parse_prune_heads


#########################
# RUNNER CONFIGURATIONS #
#########################
def get_runner_args():
    
    parser = argparse.ArgumentParser(description='Argument Parser for the S3PLR project.')
    
    # setting
    parser.add_argument('--config', default='../config/deprecated_runner/tera_libri_fmllrBase_pretrain,yaml', type=str, help='Path to experiment config.', required=False)
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)

    # Logging
    parser.add_argument('--logdir', default='../log/log_transformer/', type=str, help='Logging path.', required=False)
    parser.add_argument('--name', default=None, type=str, help='Name for logging.', required=False)

    # model ckpt
    parser.add_argument('--load', action='store_true', help='Load pre-trained model to restore training, no need to specify this during testing.')
    parser.add_argument('--ckpdir', default='../result/result_transformer/', type=str, help='path to store experiment result.', required=False)
    parser.add_argument('--ckpt', default='fmllrBase960-F-N-K-libri/states-1000000.ckpt', type=str, help='path to transformer model checkpoint.', required=False)
    parser.add_argument('--dckpt', default='baseline_sentiment_libri_sd1337/baseline_sentiment-500000.ckpt', type=str, help='path to downstream checkpoint.', required=False)
    parser.add_argument('--apc_path', default='../result/result_apc/apc_libri_sd1337_standard/apc-500000.ckpt', type=str, help='path to the apc model checkpoint.', required=False)

    # mockingjay
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--run_transformer', action='store_true', help='train and test the downstream tasks using speech representations.')
    parser.add_argument('--run_apc', action='store_true', help='train and test the downstream tasks using apc representations.')
    parser.add_argument('--fine_tune', action='store_true', help='fine tune the transformer model with downstream task.')
    parser.add_argument('--plot', action='store_true', help='Plot model generated results during testing.')
    
    # phone task
    parser.add_argument('--train_phone', action='store_true', help='Train the phone classifier on mel or speech representations.')
    parser.add_argument('--test_phone', action='store_true', help='Test mel or speech representations using the trained phone classifier.')

    # cpc phone task
    parser.add_argument('--train_cpc_phone', action='store_true', help='Train the phone classifier on mel or speech representations with the alignments in CPC paper.')
    parser.add_argument('--test_cpc_phone', action='store_true', help='Test mel or speech representations using the trained phone classifier with the alignments in CPC paper.')

    # sentiment task
    parser.add_argument('--train_sentiment', action='store_true', help='Train the sentiment classifier on mel or speech representations.')
    parser.add_argument('--test_sentiment', action='store_true', help='Test mel or speech representations using the trained sentiment classifier.')
    
    # speaker verification task
    parser.add_argument('--train_speaker', action='store_true', help='Train the speaker classifier on mel or speech representations.')
    parser.add_argument('--test_speaker', action='store_true', help='Test mel or speech representations using the trained speaker classifier.')
    
    # Options
    parser.add_argument('--with_head', action='store_true', help='inference with the spectrogram head, the model outputs spectrogram.')
    parser.add_argument('--plot_attention', action='store_true', help='plot attention')
    parser.add_argument('--load_ws', default='result/result_transformer_sentiment/10111754-10170300-weight_sum/best_val.ckpt', help='load weighted-sum weights from trained downstream model')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable Multi-GPU training.')
    parser.add_argument('--no_msg', action='store_true', help='Hide all messages.')
    parser.add_argument('--test_reconstruct', action='store_true', help='Test reconstruction capability')

    # parse
    args = parser.parse_args()
    setattr(args,'gpu', not args.cpu)
    setattr(args,'verbose', not args.no_msg)
    config = yaml.load(open(args.config,'r'), Loader=yaml.FullLoader)
    parse_prune_heads(config)
    
    return config, args


########
# MAIN #
########
def main():
    
    # get arguments
    config, args = get_runner_args()
    
    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Train Transformer
    if args.train:
        from transformer.solver import Trainer
        trainer = Trainer(config, args)
        trainer.load_data(split='train')
        trainer.set_model(inference=False)
        trainer.exec()

    # Test Transformer
    if args.test_reconstruct:
        from transformer.solver import Trainer
        trainer = Trainer(config, args)
        trainer.load_data(split='test')
        trainer.set_model(inference=True, with_head=True)
        trainer.test_reconstruct()

    ##################################################################################
    
    # Train Phone Task
    elif args.train_phone:
        from downstream.solver import Downstream_Trainer
        task = 'transformer_phone' if args.run_transformer \
                else 'apc_phone' if args.run_apc else 'baseline_phone'
        trainer = Downstream_Trainer(config, args, task=task)
        trainer.load_data(split='train', load='montreal_phone')
        trainer.set_model(inference=False)
        trainer.exec()

    # Test Phone Task
    elif args.test_phone:
        from downstream.solver import Downstream_Tester
        task = 'transformer_phone' if args.run_transformer \
                else 'apc_phone' if args.run_apc else 'baseline_phone'
        tester = Downstream_Tester(config, args, task=task)
        tester.load_data(split='test', load='montreal_phone')
        tester.set_model(inference=True)
        tester.exec()

    ##################################################################################

    # Train the CPC Phone Task
    elif args.train_cpc_phone:
        from downstream.solver import Downstream_Trainer
        task = 'transformer_cpc_phone' if args.run_transformer \
                else 'apc_cpc_phone' if args.run_apc else 'baseline_cpc_phone'
        trainer = Downstream_Trainer(config, args, task=task)
        trainer.load_data(split='train', load='cpc_phone')
        trainer.set_model(inference=False)
        trainer.exec()

    # Test Phone Task
    elif args.test_cpc_phone:
        from downstream.solver import Downstream_Tester
        task = 'transformer_cpc_phone' if args.run_transformer \
                else 'apc_cpc_phone' if args.run_apc else 'baseline_cpc_phone'
        tester = Downstream_Tester(config, args, task=task)
        tester.load_data(split='test', load='cpc_phone')
        tester.set_model(inference=True)
        tester.exec()

    ##################################################################################    

    # Train Sentiment Task
    elif args.train_sentiment:
        from downstream.solver import Downstream_Trainer
        task = 'transformer_sentiment' if args.run_transformer \
                else 'apc_sentiment' if args.run_apc else 'baseline_sentiment'
        trainer = Downstream_Trainer(config, args, task=task)
        trainer.load_data(split='train', load='sentiment')
        trainer.set_model(inference=False)
        trainer.exec()

    # Test Sentiment Task
    elif args.test_sentiment:
        from downstream.solver import Downstream_Tester
        task = 'transformer_sentiment' if args.run_transformer \
                else 'apc_sentiment' if args.run_apc else 'baseline_sentiment'
        tester = Downstream_Tester(config, args, task=task)
        tester.load_data(split='test', load='sentiment')
        tester.set_model(inference=True)
        tester.exec()

    ##################################################################################
    
    # Train Speaker Task
    elif args.train_speaker:
        from downstream.solver import Downstream_Trainer
        task = 'transformer_speaker' if args.run_transformer \
                else 'apc_speaker' if args.run_apc else 'baseline_speaker'
        trainer = Downstream_Trainer(config, args, task=task)
        trainer.load_data(split='train', load='speaker')
        # trainer.load_data(split='train', load='speaker_large') # Deprecated
        trainer.set_model(inference=False)
        trainer.exec()

    # Test Speaker Task
    elif args.test_speaker:
        from downstream.solver import Downstream_Tester
        task = 'transformer_speaker' if args.run_transformer \
                else 'apc_speaker' if args.run_apc else 'baseline_speaker'
        tester = Downstream_Tester(config, args, task=task)
        tester.load_data(split='test', load='speaker')
        # tester.load_data(split='test', load='speaker_large') # Deprecated
        tester.set_model(inference=True)
        tester.exec()

    ##################################################################################

    # Visualize Transformer
    elif args.plot:
        from transformer.solver import Tester
        tester = Tester(config, args)
        tester.load_data(split='test', load_mel_only=True)
        tester.set_model(inference=True, with_head=args.with_head)
        tester.plot(with_head=args.with_head)

    elif args.plot_attention:
        from transformer.solver import Tester
        tester = Tester(config, args)
        tester.load_data(split='test', load_mel_only=True)
        tester.set_model(inference=True, output_attention=True)
        tester.plot_attention()


if __name__ == '__main__':
    main()