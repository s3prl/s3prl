import os
import math
import torch
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .bin.train_asr import Solver

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension

            **kwargs: dict
                The arguments specified by the argparser in run_benchmark.py
                in case you need it.
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim

        paras = self._pseudo_default_args()
        config = downstream_expert

        self.solver = Solver(config, paras, mode='train')


    def _pseudo_default_args(self):
        # Only for driving Solver in bin.train_asr
        parser = argparse.ArgumentParser(description='Training E2E asr.')
        parser.add_argument('--config', type=str, help='Path to experiment config.')
        parser.add_argument('--name', default='dummy', type=str, help='Name for logging.')
        parser.add_argument('--logdir', default=f'{os.path.dirname(__file__)}/log/', type=str, help='Logging path.', required=False)
        parser.add_argument('--ckpdir', default=f'{os.path.dirname(__file__)}/ckpt/', type=str, help='Checkpoint path.', required=False)
        parser.add_argument('--outdir', default=f'{os.path.dirname(__file__)}/result/', type=str, help='Decode output path.', required=False)
        parser.add_argument('--load', default=None, type=str, help='Load pre-trained model (for training only)', required=False)
        parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducable results.', required=False)
        parser.add_argument('--cudnn-ctc', action='store_true', help='Switches CTC backend from torch to cudnn')
        parser.add_argument('--njobs', default=16, type=int, help='Number of threads for dataloader/decoding.', required=False)
        parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
        parser.add_argument('--no-pin', action='store_true', help='Disable pin-memory for dataloader')
        parser.add_argument('--test', action='store_true', help='Test the model.')
        parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
        parser.add_argument('--lm', action='store_true', help='Option for training RNNLM.')
        parser.add_argument('--amp', action='store_true', help='Option to enable AMP.')
        parser.add_argument('--reserve_gpu', default=0, type=float, help='Option to reserve GPU ram for training.')
        parser.add_argument('--jit', action='store_true', help='Option for enabling jit in pytorch. (feature in development)')
        parser.add_argument('--cuda', default=0, type=int, help='Choose which gpu to use.')

        paras = parser.parse_args([])
        setattr(paras,'gpu',not paras.cpu)
        setattr(paras,'pin_memory',not paras.no_pin)
        setattr(paras,'verbose',not paras.no_msg)
        return paras


    """
    Datalaoder Specs:
        Each dataloader should output a list in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with:
            1. dim() == 1
            2. sample_rate == 16000
            3. directly loaded by torchaudio without any preprocessing
    """

    # Interface
    def get_train_dataloader(self):
        return self.solver.get_train_dataloader()

    # Interface
    def get_dev_dataloader(self):
        return self.solver.get_dev_dataloader()

    # Interface
    def get_test_dataloader(self):
        return self.solver.get_test_dataloader()

    # Interface
    def forward(self, *args, **kwargs):
        return self.solver.forward(*args, **kwargs)

    # interface
    def log_records(self, *args, **kwargs):
        return self.solver.log_records(*args, **kwargs)
