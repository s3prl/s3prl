# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ runner_asr.py ]
#   Synopsis     [ train / test / eval of asr model ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference    [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import yaml
import torch
import random
import argparse
import numpy as np
import pandas as pd
import editdistance as ed
torch.backends.cudnn.deterministic = True
# Make cudnn CTC deterministic
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


##########################
# E2E ASR CONFIGURATIONS #
##########################
def get_asr_args():
    
    parser = argparse.ArgumentParser(description='Training E2E asr.')
    
    parser.add_argument('--config', default='config/asr_libri.yaml', type=str, help='Path to experiment config.')
    parser.add_argument('--logdir', default='log/log_asr/', type=str, help='Logging path.', required=False)
    parser.add_argument('--ckpdir', default='result/result_asr/', type=str, help='Checkpoint/Result path.', required=False)

    parser.add_argument('--name', default=None, type=str, help='Name for logging.')
    parser.add_argument('--load', default=None, type=str, help='Load pre-trained model', required=False)
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)
    parser.add_argument('--njobs', default=1, type=int, help='Number of threads for decoding.', required=False)

    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--test', action='store_true', help='Test the model.')
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
    parser.add_argument('--rnnlm', action='store_true', help='Option for training RNNLM.')

    parser.add_argument('--eval', action='store_true', help='Eval the model on test results.')
    parser.add_argument('--file', type=str, help='Path to decode result file.')
    args = parser.parse_args()

    setattr(args,'gpu',not args.cpu)
    setattr(args,'verbose',not args.no_msg)
    config = yaml.load(open(args.config,'r'))
    
    return config, args


########
# MAIN #
########
def main():
    
    # get arguments
    config, args = get_asr_args()
    
    # Train / Test
    if not args.eval:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

        if not args.rnnlm:
            if not args.test:
                # Train ASR
                from asr.solver import Trainer as Solver
            else:
                # Test ASR
                from asr.solver import Tester as Solver
        else:
            # Train RNNLM
            from asr.solver import RNNLM_Trainer as Solver

        solver = Solver(config, args)
        solver.load_data()
        solver.set_model()
        solver.exec()

    # Eval
    else:                       
        decode = pd.read_csv(args.file,sep='\t',header=None)
        truth = decode[0].tolist()
        pred = decode[1].tolist()
        cer = []
        wer = []
        for gt,pd in zip(truth,pred):
            wer.append(ed.eval(pd.split(' '),gt.split(' '))/len(gt.split(' ')))
            cer.append(ed.eval(pd,gt)/len(gt))

        print('CER : {:.6f}'.format(sum(cer)/len(cer)))
        print('WER : {:.6f}'.format(sum(wer)/len(wer)))
        print('p.s. for phoneme sequences, WER=Phone Error Rate and CER is meaningless.')


if __name__ == '__main__':
    main()

