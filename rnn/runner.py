# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ rnn/runner.py ]
#   Synopsis     [ run train / test for the apc model ]
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
from rnn.solver import Solver


######################
# APC CONFIGURATIONS #
######################
def get_runner_args():
    
    parser = argparse.ArgumentParser(description='Argument Parser for the apc model.')
    
    # mode
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--test', action='store_true', help='Test the model.')

    # setting
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)

    args = parser.parse_args()
    return args


######################
# APC CONFIGURATIONS #
######################
class get_apc_config():
    def __init__(self, seed=1337):
        # Prenet architecture (note that all APC models in the paper DO NOT incoporate a prenet)
        self.prenet_num_layers = 0 # Number of ReLU layers as prenet
        self.prenet_dropout = 0.0 # Dropout for prenet

        # RNN architecture
        self.rnn_num_layers = 3 # Number of RNN layers in the APC model
        self.rnn_hidden_size = 512 # Number of hidden units in each RNN layer, set identical to mockingjay `hidden_size`
        self.rnn_dropout = 0.1 # Dropout for each RNN output layer except the last one
        self.rnn_residual = True # Apply residual connections between RNN layers if specified

        # Training configuration
        self.optimizer = 'adam' # The gradient descent optimizer (e.g., sgd, adam, etc.)
        self.batch_size = 32 # Training minibatch size
        self.learning_rate = 0.001 # Initial learning rate
        self.total_steps = 500000 # Number of training steps
        self.time_shift = 3 # Given f_{t}, predict f_{t + n}, where n is the time_shift, , sweet spot == 3 as reported in the paper
        self.clip_thresh = 1.0 # Threshold for clipping the gradients
        self.log_step = 2500 # Log training every this amount of training steps
        self.max_keep = 2 # Maximum number of model ckpt to keep during training
        self.save_step = 10000 # Save model every this amount of training steps

        # Misc configurations
        self.feature_dim = 80 # The dimension of the input frame
        self.load_data_workers = 8 # Number of parallel data loaders
        self.experiment_name = 'apc_libri_sd' + str(seed) # Name of this experiment
        self.log_path = './result/result_apc/' # Where to save the logs
        self.result_path = './result/result_apc/' # Where to save the trained models

        # Data path configurations
        self.data_path = 'data/libri_fbank_cmvn' # Path to the preprocessed librispeech directory 
        self.train_set = ['train-clean-100'] 
        self.dev_set = ['dev-clean'] 
        self.test_set = ['test-clean']


##################
# GET APC SOLVER #
##################
def Runner(seed, train=True):
    solver = Solver(get_apc_config(seed))
    solver.load_data(split='train' if train else 'test')
    solver.set_model(inference=False if train else True)
    return solver


#################
# GET APC MODEL #
#################
def get_apc_model(path):
    solver = Solver(get_apc_config())
    solver.set_model(inference=True)
    solver.load_model(path)
    return solver


########
# MAIN #
########
def main():
    
    args = get_runner_args()

    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Train apc
    if args.train:
        solver = Runner(args.seed, train=True)
        solver.train()

    ##################################################################################

    # Test apc
    elif args.test:
        solver = Runner(args.seed, train=False)
        solver.test()


if __name__ == '__main__':
    main()