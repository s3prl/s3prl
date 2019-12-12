# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ runner_mockingjay.py ]
#   Synopsis     [ runner for the mockingjay model ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import yaml
import torch
import random
import argparse
import numpy as np
from utility.timer import Timer


#############################
# MOCKINGJAY CONFIGURATIONS #
#############################
def get_mockingjay_args():
    
    parser = argparse.ArgumentParser(description='Argument Parser for the mockingjay project.')
    
    # setting
    parser.add_argument('--config', default='config/mockingjay_libri.yaml', type=str, help='Path to experiment config.')
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)

    # Logging
    parser.add_argument('--logdir', default='log/log_mockingjay/', type=str, help='Logging path.', required=False)
    parser.add_argument('--name', default=None, type=str, help='Name for logging.', required=False)

    # model ckpt
    parser.add_argument('--load', action='store_true', help='Load pre-trained model to restore training, no need to specify this during testing.')
    parser.add_argument('--ckpdir', default='result/result_mockingjay/', type=str, help='Checkpoint/Result path.', required=False)
    parser.add_argument('--ckpt', default='mockingjay_libri_sd1337_LinearLarge/mockingjay-500000.ckpt', type=str, help='path to mockingjay model checkpoint.', required=False)
    # parser.add_argument('--ckpt', default='mockingjay_libri_sd1337_MelBase/mockingjay-500000.ckpt', type=str, help='path to mockingjay model checkpoint.', required=False)
    parser.add_argument('--dckpt', default='baseline_sentiment_libri_sd1337/baseline_sentiment-500000.ckpt', type=str, help='path to downstream checkpoint.', required=False)
    parser.add_argument('--apc_path', default='./result/result_apc/apc_libri_sd1337_standard/apc-500000.ckpt', type=str, help='path to the apc model checkpoint.', required=False)

    # mockingjay
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--run_mockingjay', action='store_true', help='train and test the downstream tasks using mockingjay representations.')
    parser.add_argument('--run_apc', action='store_true', help='train and test the downstream tasks using apc representations.')
    parser.add_argument('--fine_tune', action='store_true', help='fine tune the mockingjay model with downstream task.')
    parser.add_argument('--plot', action='store_true', help='Plot model generated results during testing.')
    
    # phone task
    parser.add_argument('--train_phone', action='store_true', help='Train the phone classifier on mel or mockingjay representations.')
    parser.add_argument('--test_phone', action='store_true', help='Test mel or mockingjay representations using the trained phone classifier.')
    
    # sentiment task
    parser.add_argument('--train_sentiment', action='store_true', help='Train the sentiment classifier on mel or mockingjay representations.')
    parser.add_argument('--test_sentiment', action='store_true', help='Test mel or mockingjay representations using the trained sentiment classifier.')
    
    # speaker verification task
    parser.add_argument('--train_speaker', action='store_true', help='Train the speaker classifier on mel or mockingjay representations.')
    parser.add_argument('--test_speaker', action='store_true', help='Test mel or mockingjay representations using the trained speaker classifier.')
    
    # Options
    parser.add_argument('--with_head', action='store_true', help='inference with the spectrogram head, the model outputs spectrogram.')
    parser.add_argument('--output_attention', action='store_true', help='plot attention')
    parser.add_argument('--load_ws', default='result/result_mockingjay_sentiment/10111754-10170300-weight_sum/best_val.ckpt', help='load weighted-sum weights from trained downstream model')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')


    args = parser.parse_args()
    setattr(args,'gpu', not args.cpu)
    setattr(args,'verbose', not args.no_msg)
    config = yaml.load(open(args.config,'r'))
    config['timer'] = Timer()
    
    return config, args


########
# MAIN #
########
def main():
    
    # get arguments
    config, args = get_mockingjay_args()
    
    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Train Mockingjay
    if args.train:
        from mockingjay.solver import Trainer
        trainer = Trainer(config, args)
        trainer.load_data(split='train')
        trainer.set_model(inference=False)
        trainer.exec()

    ##################################################################################
    
    # Train Phone Task
    elif args.train_phone:
        from downstream.solver import Downstream_Trainer
        task = 'mockingjay_phone' if args.run_mockingjay \
                else 'apc_phone' if args.run_apc else 'baseline_phone'
        trainer = Downstream_Trainer(config, args, task=task)
        trainer.load_data(split='train', load='phone')
        trainer.set_model(inference=False)
        trainer.exec()

    # Test Phone Task
    elif args.test_phone:
        from downstream.solver import Downstream_Tester
        task = 'mockingjay_phone' if args.run_mockingjay \
                else 'apc_phone' if args.run_apc else 'baseline_phone'
        tester = Downstream_Tester(config, args, task=task)
        tester.load_data(split='test', load='phone')
        tester.set_model(inference=True)
        tester.exec()

    ##################################################################################

    # Train Sentiment Task
    elif args.train_sentiment:
        from downstream.solver import Downstream_Trainer
        task = 'mockingjay_sentiment' if args.run_mockingjay \
                else 'apc_sentiment' if args.run_apc else 'baseline_sentiment'
        trainer = Downstream_Trainer(config, args, task=task)
        trainer.load_data(split='train', load='sentiment')
        trainer.set_model(inference=False)
        trainer.exec()

    # Test Sentiment Task
    elif args.test_sentiment:
        from downstream.solver import Downstream_Tester
        task = 'mockingjay_sentiment' if args.run_mockingjay \
                else 'apc_sentiment' if args.run_apc else 'baseline_sentiment'
        tester = Downstream_Tester(config, args, task=task)
        tester.load_data(split='test', load='sentiment')
        tester.set_model(inference=True)
        tester.exec()

    ##################################################################################
    
    # Train Speaker Task
    elif args.train_speaker:
        from downstream.solver import Downstream_Trainer
        task = 'mockingjay_speaker' if args.run_mockingjay \
                else 'apc_speaker' if args.run_apc else 'baseline_speaker'
        trainer = Downstream_Trainer(config, args, task=task)
        trainer.load_data(split='train', load='speaker')
        # trainer.load_data(split='train', load='speaker_large') # Deprecated
        trainer.set_model(inference=False)
        trainer.exec()

    # Test Speaker Task
    elif args.test_speaker:
        from downstream.solver import Downstream_Tester
        task = 'mockingjay_speaker' if args.run_mockingjay \
                else 'apc_speaker' if args.run_apc else 'baseline_speaker'
        tester = Downstream_Tester(config, args, task=task)
        tester.load_data(split='test', load='speaker')
        # tester.load_data(split='test', load='speaker_large') # Deprecated
        tester.set_model(inference=True)
        tester.exec()

    ##################################################################################

    # Visualize Mockingjay
    elif args.plot:
        from mockingjay.solver import Tester
        tester = Tester(config, args)
        tester.load_data(split='test', load_mel_only=True)
        tester.set_model(inference=True, with_head=args.with_head, output_attention=args.output_attention)
        tester.plot(with_head=args.with_head)

    config['timer'].report()


########################
# GET MOCKINGJAY MODEL #
########################
def get_mockingjay_model(from_path='result/result_mockingjay/mockingjay_libri_sd1337_best/mockingjay-500000.ckpt', display_settings=False):
    ''' Wrapper that loads the mockingjay model from checkpoint path '''

    # load config and paras
    all_states = torch.load(from_path, map_location='cpu')
    config = all_states['Settings']['Config']
    paras = all_states['Settings']['Paras']

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
    from mockingjay.solver import Tester
    mockingjay = Tester(config, paras)
    mockingjay.set_model(inference=True, with_head=False, from_path=from_path)
    return mockingjay


if __name__ == '__main__':
    main()

