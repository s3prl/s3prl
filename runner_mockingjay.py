# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ runner_mockingjay.py ]
#   Synopsis     [ training for the mockingjay model ]
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
import pandas as pd
from mockingjay.solver import Trainer as Solver


# Make cudnn CTC deterministic
torch.backends.cudnn.deterministic = True
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning) 


#############################
# MOCKINGJAY CONFIGURATIONS #
#############################
def get_mockingjay_args():
	
	parser = argparse.ArgumentParser(description='Training E2E asr.')
	
	parser.add_argument('--config', default='config/mockingjay_libri.yaml', type=str, help='Path to experiment config.')
	parser.add_argument('--logdir', default='log_mockingjay/', type=str, help='Logging path.', required=False)
	parser.add_argument('--ckpdir', default='result_mockingjay/', type=str, help='Checkpoint/Result path.', required=False)

	parser.add_argument('--name', default=None, type=str, help='Name for logging.', required=False)
	parser.add_argument('--load', default=None, type=str, help='Load pre-trained model', required=False)
	parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)
	# parser.add_argument('--njobs', default=1, type=int, help='Number of threads for decoding.', required=False)

	parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
	# parser.add_argument('--test', action='store_true', help='Test the model.')
	parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
	# parser.add_argument('--rnnlm', action='store_true', help='Option for training RNNLM.')

	# parser.add_argument('--eval', action='store_true', help='Eval the model on test results.')
	# parser.add_argument('--file', type=str, help='Path to decode result file.')
	args = parser.parse_args()

	setattr(args,'gpu', not args.cpu)
	setattr(args,'verbose', not args.no_msg)
	config = yaml.load(open(args.config,'r'))
	
	return config, args


########
# MAIN #
########
def main():
	
	# get arguments
	config, args = get_mockingjay_args()
	
	# Train
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

	solver = Solver(config, args)
	solver.load_data()
	solver.set_model()
	solver.exec()


if __name__ == '__main__':
	main()

