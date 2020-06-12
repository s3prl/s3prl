# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ rnn/solver.py ]
#   Synopsis     [ solver for performing pre-training and testing of rnn models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference    [ Modified and rewrite based on: https://github.com/iamyuanchung/Autoregressive-Predictive-Coding ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import namedtuple
from torch import nn, optim
from torch.utils import data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from dataloader import get_Dataloader
from rnn.model import APCModel
from utility.audio import plot_spectrogram_to_numpy


PrenetConfig = namedtuple(
    'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])
RNNConfig = namedtuple(
    'RNNConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout', 'residual'])


class Solver():

    def __init__(self, config):
        
        self.config = config
        self.model_dir = os.path.join(self.config.result_path, self.config.experiment_name)
        self.log_dir = os.path.join(self.config.log_path, self.config.experiment_name)


    def verbose(self, msg, end='\n'):
        ''' Verbose function for print information to stdout'''
        print('[SOLVER] - ', msg, end=end)


    def load_data(self, split='train'):
        ''' Load data for training / testing'''
        if split == 'train': 
            self.verbose('Loading source data from ' + str(self.config.train_set) + ' from ' + self.config.data_path)
        elif split == 'test': 
            self.verbose('Loading testing data ' + str(self.config.test_set) + ' from ' + self.config.data_path)
        else:
            raise NotImplementedError('Invalid `split` argument!')
        setattr(self, 'dataloader', get_Dataloader(split, load='acoustic', data_path=self.config.data_path, 
                                                   batch_size=self.config.batch_size, 
                                                   max_timestep=3000, max_label_len=400, 
                                                   use_gpu=True, n_jobs=self.config.load_data_workers, 
                                                   train_set=self.config.train_set, 
                                                   dev_set=self.config.dev_set, 
                                                   test_set=self.config.test_set, 
                                                   dev_batch_size=1))


    def set_model(self, inference=False):
        if self.config.prenet_num_layers == 0:
            prenet_config = None
            rnn_config = RNNConfig(
                self.config.feature_dim, self.config.rnn_hidden_size, self.config.rnn_num_layers,
                self.config.rnn_dropout, self.config.rnn_residual)
        else:
            prenet_config = PrenetConfig(
                self.config.feature_dim, self.config.rnn_hidden_size, self.config.prenet_num_layers,
                self.config.prenet_dropout)
            rnn_config = RNNConfig(
                self.config.rnn_hidden_size, self.config.rnn_hidden_size, self.config.rnn_num_layers,
                self.config.rnn_dropout, self.config.rnn_residual)

        self.model = APCModel(mel_dim=self.config.feature_dim,
                              prenet_config=prenet_config,
                              rnn_config=rnn_config).cuda()

        if not inference:
            self.criterion = nn.L1Loss()
            if self.config.optimizer == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            elif self.config.optimizer == 'adadelta':
                self.optimizer = optim.Adadelta(self.model.parameterlearning_rates())
            elif self.config.optimizer == 'sgd':
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
            elif self.config.optimizer == 'adagrad':
                self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.config.learning_rate)
            elif self.config.optimizer == 'rmsprop':
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate)
            else:
                raise NotImplementedError("Learning method not supported for the task")
            self.log = SummaryWriter(self.log_dir)


    def load_model(self, path):
        self.verbose('Load model from {}'.format(path))
        state = torch.load(path, map_location='cpu')
        try:
            self.model.load_state_dict(state)
            self.verbose('[APC] - Loaded')
        except: self.verbose('[APC - X]')


    def process_data(self, batch_x):
        assert(len(batch_x.shape) == 4), 'Bucketing should cause acoustic feature to have shape 1xBxTxD'
        with torch.no_grad():
            # Hack bucket
            batch_x = batch_x.squeeze(0)
            # compute length for each uttr
            batch_l = np.sum(np.sum(batch_x.data.numpy(), axis=-1) != 0, axis=-1)
            batch_l = [int(sl) for sl in batch_l]
        return torch.FloatTensor(batch_x), torch.FloatTensor(batch_l)


    ####################
    ##### Training #####
    ####################
    def train(self):
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model.train()
        pbar = tqdm(total=self.config.total_steps)
        
        model_kept = []
        global_step = 1

        while global_step <= self.config.total_steps:

            for batch_x in tqdm(self.dataloader, desc="Iteration"):
                if global_step > self.config.total_steps: break

                batch_x, batch_l = self.process_data(batch_x)
                _, indices = torch.sort(batch_l, descending=True)

                batch_x = Variable(batch_x[indices]).cuda()
                batch_l = Variable(batch_l[indices]).cuda()

                outputs, _ = self.model(batch_x[:, :-self.config.time_shift, :], \
                                        batch_l - self.config.time_shift)

                loss = self.criterion(outputs, batch_x[:, self.config.time_shift:, :])
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_thresh)
                
                # Step
                if math.isnan(grad_norm):
                    print('Error : grad norm is NaN @ step ' + str(global_step))
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if global_step % self.config.log_step == 0:
                    self.log.add_scalar('training loss (step-wise)', float(loss.item()), global_step)
                    self.log.add_scalar('gradient norm', grad_norm, global_step)

                # log and save
                if global_step % self.config.save_step == 0:
                    pred_spec = plot_spectrogram_to_numpy(outputs[0].data.cpu().numpy())
                    true_spec = plot_spectrogram_to_numpy(batch_x[0].data.cpu().numpy())
                    self.log.add_image('pred_spec', pred_spec, global_step)
                    self.log.add_image('true_spec', true_spec, global_step)
                    new_model_path = os.path.join(self.model_dir, 'apc-%d' % global_step + '.ckpt')
                    torch.save(self.model.state_dict(), new_model_path)
                    model_kept.append(new_model_path)

                    if len(model_kept) > self.config.max_keep:
                        os.remove(model_kept[0])
                        model_kept.pop(0)

                pbar.update(1)
                global_step += 1


    ###################
    ##### Testing #####
    ###################
    def test(self):
        self.model.eval()
        test_losses = []
        with torch.no_grad():
            for test_batch_x in self.dataloader:

                test_batch_x, test_batch_l = self.process_data(test_batch_x)
                _, test_indices = torch.sort(test_batch_l, descending=True)

                test_batch_x = Variable(test_batch_x[test_indices]).cuda()
                test_batch_l = Variable(test_batch_l[test_indices]).cuda()

                test_outputs, _ = self.model(test_batch_x[:, :-self.config.time_shift, :], \
                                             test_batch_l - self.config.time_shift)

                test_loss = self.criterion(test_outputs, test_batch_x[:, self.config.time_shift:, :])
                test_losses.append(test_loss.item())


    ###################
    ##### Forward #####
    ###################
    def forward(self, batch_x, all_layers=True):
        self.model.eval()
        with torch.no_grad():

            batch_x, batch_l = self.process_data(batch_x)
            _, indices = torch.sort(batch_l, descending=True)

            batch_x = Variable(batch_x[indices]).cuda()
            batch_l = Variable(batch_l[indices]).cuda()

            _, feats = self.model(batch_x, batch_l)
        # feats shape: (num_layers, batch_size, seq_len, rnn_hidden_size)
        if not all_layers:
            return feats[-1, :, :, :] # (batch_size, seq_len, rnn_hidden_size)
        else:
            return feats.permute(1, 0, 2, 3).contiguous() # (batch_size, num_layers, seq_len, rnn_hidden_size)