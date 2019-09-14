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


#############################
# MOCKINGJAY CONFIGURATIONS #
#############################
def get_mockingjay_args():
	
	parser = argparse.ArgumentParser(description='Training E2E asr.')
	
	# setting
	parser.add_argument('--config', default='config/mockingjay_libri.yaml', type=str, help='Path to experiment config.')
	parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)

	# Logging
	parser.add_argument('--logdir', default='log_mockingjay/', type=str, help='Logging path.', required=False)
	parser.add_argument('--name', default=None, type=str, help='Name for logging.', required=False)

	# model ckpt
	parser.add_argument('--load', action='store_true', help='Load pre-trained model')
	parser.add_argument('--ckpdir', default='result_mockingjay/', type=str, help='Checkpoint/Result path.', required=False)
	parser.add_argument('--ckpt', default='mockingjay_libri_sd1337_0908/mockingjay-789600.ckpt', type=str, help='path to model checkpoint', required=False)
	# parser.add_argument('--njobs', default=1, type=int, help='Number of threads for decoding.', required=False)

	# modes
	parser.add_argument('--train', action='store_true', help='Test the model.')
	parser.add_argument('--train_phone', action='store_true', help='Train the phone classifier on mockingjay representations.')
	parser.add_argument('--test_phone', action='store_true', help='Test mockingjay representations using the trained phone classifier.')
	parser.add_argument('--plot', action='store_true', help='Plot model generated results during testing.')
	
	# Options
	parser.add_argument('--with-head', action='store_true', help='inference with the spectrogram head, the model outputs spectrogram.')
	parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
	parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')


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
	
	# Fix seed and make backends deterministic
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True

	if args.train:
		from mockingjay.solver import Trainer
		trainer = Trainer(config, args)
		trainer.load_data(dataset='train')
		trainer.set_model(inference=False)
		trainer.train()

	elif args.test_phone:
		from mockingjay.solver import Tester
		tester = Tester(config, args)
		tester.load_data(dataset='test', phone_loader=True)
		tester.set_model(inference=True, with_head=False)
		tester.test_phone()

	elif args.plot:
		from mockingjay.solver import Tester
		tester = Tester(config, args)
		tester.load_data(dataset='test', phone_loader=False)
		tester.set_model(inference=True, with_head=args.with_head)
		tester.plot(with_head=args.with_head)


if __name__ == '__main__':
	main()

