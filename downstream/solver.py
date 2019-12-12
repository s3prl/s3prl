# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ downstream/solver.py ]
#   Synopsis     [ solvers for the mockingjay downstream model: trainer / tester ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import copy
import math
import random
import librosa
import numpy as np
from torch.optim import Adam
from tqdm import tqdm, trange
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataloader import get_Dataloader
from mockingjay.solver import Solver, Tester
from mockingjay.optimization import BertAdam
from downstream.model import LinearClassifier, RnnClassifier
from utility.audio import mel_dim, num_freq, sample_rate, inv_spectrogram
from utility.timer import Timer
from runner_apc import get_apc_model


##########
# SOLVER #
##########
class Downstream_Solver(Solver):
    ''' Handler for complete training progress'''
    def __init__(self, config, paras, task):
        super(Downstream_Solver, self).__init__(config, paras)
        # Downstream task the solver is solving
        self.task = task
        self.mock_paras = copy.deepcopy(paras)
        self.mock_config = copy.deepcopy(config)
        self.mock_config['timer'] = config['timer']

        # path and directories
        self.exp_name = self.exp_name.replace('mockingjay', task)
        self.paras.ckpdir = paras.ckpdir.replace('mockingjay', task)
        self.ckpdir = self.ckpdir.replace('mockingjay', task)
        self.ckpt = os.path.join(paras.ckpdir, paras.dckpt)

        # modify log directory
        paras.logdir = paras.logdir.replace('mockingjay', task)

        # model
        self.load_model_list = config['downstream']['load_model_list']
        self.fine_tune = paras.fine_tune
        self.run_mockingjay = True if 'mockingjay' in task else False
        self.run_apc = True if 'apc' in task else False
        if self.fine_tune:  
            assert(self.run_mockingjay), 'Use `--run_mockingjay` to fine-tune the mockingjay model.'
            assert(not self.run_apc), 'Fine tuning only supports the mockingjay model.'
            assert(not self.paras.with_head), 'Fine tuning only supports the mockingjay model, not with head.'
        assert( not (self.run_mockingjay and self.run_apc) ), 'Mockingjay and Apc can not run at the same time!'
        if self.run_mockingjay and self.paras.with_head: self.verbose('Using Mockingjay representations from head.')
        elif self.run_mockingjay and self.fine_tune: self.verbose('Fine-tuning on Mockingjay representations.')
        elif self.run_mockingjay: self.verbose('Using Mockingjay representations.')


    def load_data(self, split='train', load='phone'):
        ''' Load date for training / testing'''
        assert(load in ['phone', 'sentiment', 'speaker', 'speaker_large']), 'Unsupported dataloader!'
        if load == 'phone' or load == 'speaker_large':
            if split == 'train':
                self.verbose('Loading source data from ' + str(self.config['dataloader']['train_set']) + ' from ' + self.config['dataloader']['data_path'])
                if load == 'phone': self.verbose('Loading phone data from ' + str(self.config['dataloader']['train_set']) + ' from ' + self.config['dataloader']['phone_path'])
            elif split == 'test': 
                self.verbose('Loading testing data ' + str(self.config['dataloader']['test_set']) + ' from ' + self.config['dataloader']['data_path'])
                if load == 'phone': self.verbose('Loading label data ' + str(self.config['dataloader']['test_set']) + ' from ' + self.config['dataloader']['phone_path'])
            else:
                raise NotImplementedError('Invalid `split` argument!')
        elif load == 'speaker':
            if split == 'train':
                self.verbose('Loading source data from ' + str(self.config['dataloader']['train_set']).replace('360', '100') + ' from ' + self.config['dataloader']['data_path'])
            elif split == 'test':
                self.verbose('Loading testing data ' + str(self.config['dataloader']['test_set']).replace('360', '100') + ' from ' + self.config['dataloader']['data_path'])
            else:
                raise NotImplementedError('Invalid `split` argument!')
        elif load == 'sentiment':
            target = self.config['dataloader']['sentiment_config']['dataset']
            sentiment_path = self.config['dataloader']['sentiment_config'][target]['path']
            self.verbose(f'Loading {split} data from {sentiment_path}')
        else:
            raise NotImplementedError('Unsupported downstream tasks.')

        setattr(self, 'dataloader', get_Dataloader(split, load=load, use_gpu=self.paras.gpu, \
                run_mockingjay=self.run_mockingjay, mock_config=self.config['mockingjay'], \
                **self.config['dataloader']))


    def set_model(self, inference=False):
        self.model_type = 'linear' if 'phone' in self.task else 'rnn'
        input_dim = int(self.config['downstream'][self.model_type]['input_dim']) if \
                    self.config['downstream'][self.model_type]['input_dim'] != 'None' else None
        if 'mockingjay' in self.task:
            self.mockingjay = Tester(self.mock_config, self.mock_paras)
            if self.fine_tune and inference: self.mockingjay.load = False # Do not load twice when testing the fine-tuned model, load only for fine-tune training
            self.mockingjay.set_model(inference=True, with_head=self.paras.with_head)
            self.dr = self.mockingjay.dr
            if input_dim is None:
                input_dim = self.mock_config['mockingjay']['hidden_size']
        elif 'apc' in self.task:
            self.apc = get_apc_model(path=self.paras.apc_path)
            if input_dim is None: 
                input_dim = self.mock_config['mockingjay']['hidden_size'] # use identical dim size for fair comparison
        elif 'baseline' in self.task:
            if input_dim is None: 
                input_dim = mel_dim
        else:
            raise NotImplementedError('Invalid Task!')

        if self.model_type == 'linear':
            self.classifier = LinearClassifier(input_dim=input_dim,
                                               class_num=self.dataloader.dataset.class_num,
                                               task=self.task,
                                               dconfig=self.config['downstream']['linear'],
                                               sequencial=False).to(self.device)
        elif self.model_type == 'rnn':
            self.classifier = RnnClassifier(input_dim=input_dim,
                                            class_num=self.dataloader.dataset.class_num,
                                            task=self.task,
                                            dconfig=self.config['downstream']['rnn']).to(self.device)

        if not inference and self.fine_tune:
            # Setup Fine tune optimizer
            self.mockingjay.mockingjay.train()
            param_optimizer = list(self.mockingjay.mockingjay.named_parameters()) + list(self.classifier.named_parameters())
            self.optimizer = get_mockingjay_optimizer(params=param_optimizer, 
                                                      lr=self.learning_rate, 
                                                      warmup_proportion=self.config['optimizer']['warmup_proportion'],
                                                      training_steps=self.total_steps)
        elif not inference:
            self.optimizer = Adam(self.classifier.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
            self.classifier.train()
        else:
            self.classifier.eval()

        if self.load: # This will be set to True by default when Tester is running set_model()
            self.load_model(inference=inference)


    def save_model(self, name, model_all=True, assign_name=None):
        if model_all:
            all_states = {
                'Classifier': self.classifier.state_dict(),
                'Mockingjay': self.mockingjay.mockingjay.state_dict() if self.fine_tune else None,
                'Optimizer': self.optimizer.state_dict(),
                'Global_step': self.global_step,
                'Settings': {
                    'Config': self.config,
                    'Paras': self.paras,
                },
            }
        else:
            all_states = {
                'Classifier': self.classifier.state_dict(),
                'Settings': {
                    'Config': self.config,
                    'Paras': self.paras,
                },
            }

        if assign_name is not None:
            model_path = f'{self.ckpdir}/{assign_name}.ckpt'
            torch.save(all_states, model_path)
            return

        new_model_path = '{}/{}-{}.ckpt'.format(self.ckpdir, name, self.global_step)
        torch.save(all_states, new_model_path)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)


    def load_model(self, inference=False):
        self.verbose('Load model from {}'.format(self.ckpt))
        all_states = torch.load(self.ckpt, map_location='cpu')
        
        if 'Classifier' in self.load_model_list:
            try:
                self.classifier.load_state_dict(all_states['Classifier'])
                self.verbose('[Classifier] - Loaded')
            except: self.verbose('[Classifier - X]')

        if 'Optimizer' in self.load_model_list and not inference:
            try:
                self.optimizer.load_state_dict(all_states['Optimizer'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                self.verbose('[Optimizer] - Loaded')
            except: self.verbose('[Optimizer - X]')

        if 'Global_step' in self.load_model_list:
            try:
                self.global_step = all_states['Global_step']
                self.verbose('[Global_step] - Loaded')
            except: self.verbose('[Global_step - X]')

        if self.fine_tune:
            try:
                self.verbose('@ Downstream, [Fine-Tuned Mockingjay] - Loading from Upstream Tester...')
                self.mockingjay.load_model(inference=inference, from_path=self.ckpt)
                self.verbose('@ Downstream, [Fine-Tuned Mockingjay] - Loaded')
            except: self.verbose('[Fine-Tuned Mockingjay] - X')

        self.verbose('Model loading complete!')


###########
# TRAINER #
###########
class Downstream_Trainer(Downstream_Solver):
    ''' Handler for complete training progress'''
    def __init__(self, config, paras, task):
        super(Downstream_Trainer, self).__init__(config, paras, task)

        # Logger Settings
        self.logdir = os.path.join(paras.logdir, self.exp_name)
        self.log = SummaryWriter(self.logdir)

        # Training details
        self.log_step = config['downstream']['log_step']
        self.save_step = config['downstream']['save_step']
        self.dev_step = config['downstream']['dev_step']
        self.total_steps = config['downstream']['total_steps']
        self.learning_rate = float(config['downstream']['learning_rate'])
        self.max_keep = config['downstream']['max_keep']
        self.eval = config['downstream']['evaluation']
        self.gradient_clipping = config['optimizer']['gradient_clipping']
        self.reset_train()
        if self.fine_tune:
             self.total_steps = self.total_steps * 2 # train two epcohs to fine-tune the model, set steps manually in config/*.yaml

        # mkdir
        if not os.path.exists(self.paras.ckpdir): os.makedirs(self.paras.ckpdir)
        if not os.path.exists(self.ckpdir): os.makedirs(self.ckpdir)


    def reset_train(self):
        self.model_kept = []
        self.global_step = 1


    def exec(self):
        ''' Training of downstream tasks'''
        self.verbose('Training set total ' + str(len(self.dataloader)) + ' batches.')

        pbar = tqdm(total=self.total_steps)
        corrects = 0
        valids = 0
        best_acc = 0.0
        best_val_acc = 0.0
        loses = 0.0
        while self.global_step <= self.total_steps:

            for features, labels in tqdm(self.dataloader, desc="Iteration"):
                try:
                    if self.global_step > self.total_steps: break
                    # features: (1, batch_size, seq_len, feature)
                    # dimension of labels is depends on task and dataset, but the first dimention is always trivial due to bucketing
                    # eg. (1, batch_size, seq_len) or (1, batch_size)
                    labels = labels.squeeze(0).to(device=self.device)  # labels can be torch.long or torch.float (regression)
                    if 'speaker' in self.task: # Doesn't need the whole utterance to predict speaker
                        original_len = features[0].size(2)
                        reduce_factor = 3
                        if self.run_mockingjay: features = (features[0][:, :, :original_len//reduce_factor, :], features[1][:, :, :original_len//reduce_factor, :], features[2][:, :, :original_len//reduce_factor])
                        else: features = features[:, :, :original_len//reduce_factor, :]
                    if self.run_mockingjay and self.paras.with_head:
                        # representations shape: (batch_size, seq_len, feature)
                        representations = self.mockingjay.forward_with_head(features, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0))
                    elif self.run_mockingjay and self.fine_tune:
                        # representations shape: (batch_size, seq_len, feature)
                        representations = self.mockingjay.forward_fine_tune(features, tile=False if 'speaker' in self.task else True, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0)) if 'speaker' not in self.task else features[0].squeeze(0)
                    elif self.run_mockingjay:
                        # representations shape: (batch_size, layer, seq_len, feature)
                        representations = self.mockingjay.forward(features, tile=False if 'speaker' in self.task else True, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0)) if 'speaker' not in self.task else features[0].squeeze(0)
                    elif self.run_apc:
                        # representations shape: (batch_size, layer, seq_len, feature)
                        representations = self.apc.forward(features)
                        features = features.squeeze(0)
                    else:
                        # representations shape: (batch_size, seq_len, feature)
                        features = features.squeeze(0)
                        representations = features.to(device=self.device, dtype=torch.float32)

                    # Since zero padding technique, some timestamps of features are not valid
                    # For each timestamps, we mark 1 on valid timestamps, and 0 otherwise
                    # This variable can be useful for frame-wise metric, like phoneme recognition or speaker verification
                    # label_mask: (batch_size, seq_len), LongTensor
                    # valid_lengths: (batch_size), LongTensor
                    label_mask = (features.sum(dim=-1) != 0).type(torch.LongTensor).to(device=self.device, dtype=torch.long)
                    valid_lengths = label_mask.sum(dim=1)

                    if self.model_type == 'linear':
                        # labels: (batch_size, seq_len)
                        loss, _, correct, valid = self.classifier(representations, labels, label_mask)
                    elif self.model_type == 'rnn':
                        # labels: (batch_size, )
                        loss, _, correct, valid = self.classifier(representations, labels, valid_lengths)
                    else:
                        raise NotImplementedError('Invalid `model_type`!')

                    # Accumulate Loss
                    loss.backward()

                    loses += loss.detach().item()
                    corrects += correct
                    valids += valid

                    # Update
                    if self.fine_tune: 
                        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.mockingjay.mockingjay.parameters()) + list(self.classifier.parameters()), \
                                                                   self.gradient_clipping)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), \
                                                                   self.gradient_clipping)
                    if math.isnan(grad_norm):
                        self.verbose('Error : grad norm is NaN @ step ' + str(self.global_step))
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.global_step % self.log_step == 0:
                        # Log
                        acc = corrects.item() / valids.item()
                        los = loses / self.log_step
                        self.log.add_scalar('acc', acc, self.global_step)
                        self.log.add_scalar('loss', los, self.global_step)
                        self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                        pbar.set_description('Loss %.5f, Acc %.5f' % (los, acc))

                        loses = 0.0
                        corrects = 0
                        valids = 0

                    if self.global_step % self.save_step == 0 and acc > best_acc:
                        self.save_model(self.task)
                        best_acc = acc

                    if self.eval != 'None' and self.global_step % self.dev_step == 0:
                        self.save_model(self.task, assign_name='tmp')
                        torch.cuda.empty_cache()

                        evaluation = self.config['downstream']['evaluation']
                        tmp_model_path = '{}/tmp.ckpt'.format(self.ckpdir)
                        new_dckpt = '/'.join(tmp_model_path.split('/')[-2:])
                        test_config = copy.deepcopy(self.mock_config)
                        test_paras = copy.deepcopy(self.mock_paras)
                        test_paras.dckpt = new_dckpt
                        tester = Downstream_Tester(test_config, test_paras, task=self.task)
                        tester.load_data(split=evaluation, load=self.task.split('_')[-1])
                        tester.set_model(inference=True)
                        eval_loss, eval_acc, eval_logits = tester.exec()
                        self.log.add_scalar(f'{evaluation}_loss', eval_loss, self.global_step)
                        self.log.add_scalar(f'{evaluation}_acc', eval_acc, self.global_step)
                        if eval_acc > best_val_acc:
                            self.verbose('Saving new best model on validation')
                            self.save_model(self.task, assign_name='best_val')
                            torch.save(eval_logits, f'{self.ckpdir}/best_val.logits')
                            torch.cuda.empty_cache()
                            best_val_acc = eval_acc
                
                except RuntimeError:
                    print('CUDA out of memory at step: ', self.global_step)
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()

                pbar.update(1)
                self.global_step += 1
                
        pbar.close()
        self.reset_train()


##########
# TESTER #
##########
class Downstream_Tester(Downstream_Solver):
    ''' Handler for complete testing progress'''
    def __init__(self, config, paras, task):
        super(Downstream_Tester, self).__init__(config, paras, task)
        self.duo_feature = False # Set duo feature to False since only input mel is needed during testing
        self.load = True # Tester will load pre-trained models automatically
    
    def exec(self):
        ''' Testing of downstream tasks'''
        self.verbose('Testing set total ' + str(len(self.dataloader)) + ' batches.')
        timer = Timer()
        timer.start()

        valid_count = 0
        correct_count = 0
        loss_sum = 0
        all_logits = []

        oom_counter = 0
        for features, labels in tqdm(self.dataloader, desc="Iteration"):
            with torch.no_grad():
                try:
                    # features: (1, batch_size, seq_len, feature)
                    # dimension of labels is depends on task and dataset, but the first dimention is always trivial due to bucketing
                    labels = labels.squeeze(0).to(device=self.device)

                    if self.run_mockingjay and self.paras.with_head:
                        # representations shape: (batch_size, seq_len, feature)
                        representations = self.mockingjay.forward_with_head(features, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0))
                    elif self.run_mockingjay and self.fine_tune:
                        # representations shape: (batch_size, seq_len, feature)
                        representations = self.mockingjay.forward_fine_tune(features, tile=False if 'speaker' in self.task else True, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0)) if 'speaker' not in self.task else features[0].squeeze(0)
                    elif self.run_mockingjay:
                        # representations shape: (batch_size, layer, seq_len, feature)
                        representations = self.mockingjay.forward(features, tile=False if 'speaker' in self.task else True, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0)) if 'speaker' not in self.task else features[0].squeeze(0)
                    elif self.run_apc:
                        # representations shape: (batch_size, layer, seq_len, feature)
                        representations = self.apc.forward(features)
                        features = features.squeeze(0)
                    else:
                        # representations shape: (batch_size, seq_len, feature)
                        features = features.squeeze(0)
                        representations = features.to(device=self.device, dtype=torch.float32)

                    # Since zero padding technique, some timestamps of features are not valid
                    # For each timestamps, we mark 1 on valid timestamps, and 0 otherwise
                    # This variable can be useful for frame-wise metric, like phoneme recognition or speaker verification
                    # label_mask: (batch_size, seq_len), LongTensor
                    label_mask = (features.sum(dim=-1) != 0).type(torch.LongTensor).to(device=self.device, dtype=torch.long)
                    valid_lengths = label_mask.sum(dim=1)

                    if self.model_type == 'linear':
                        # labels: (batch_size, seq_len)
                        loss, logits, correct, valid = self.classifier(representations, labels, label_mask)
                    elif self.model_type == 'rnn':
                        # labels: (batch_size, )
                        loss, logits, correct, valid = self.classifier(representations, labels, valid_lengths)
                    else:
                        raise NotImplementedError
                    
                    loss_sum += loss.detach().cpu().item()
                    all_logits.append(logits)
                    correct_count += correct.item()
                    valid_count += valid.item()

                except RuntimeError:
                    if oom_counter > 10: break
                    else: oom_counter += 1
                    print('CUDA out of memory during testing, aborting after ' + str(10 - oom_counter) + ' more tries...')
                    torch.cuda.empty_cache()

        average_loss = loss_sum / len(self.dataloader)
        test_acc = correct_count * 1.0 / valid_count
        self.verbose(f'Test result: loss {average_loss}, acc {test_acc}')

        timer.end()
        timer.report()
        
        return average_loss, test_acc, all_logits


def get_mockingjay_optimizer(params, lr, warmup_proportion, training_steps):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=lr,
                         warmup=warmup_proportion,
                         t_total=training_steps)
    return optimizer
