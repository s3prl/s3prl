# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ downstream/solver.py ]
#   Synopsis     [ solvers for the transformer downstream model: trainer / tester ]
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
from transformer.solver import Solver, Tester
from transformer.optimization import BertAdam
from downstream.model import LinearClassifier, RnnClassifier
from rnn.runner import get_apc_model


##########
# SOLVER #
##########
class Downstream_Solver(Solver):
    ''' Handler for complete training progress'''
    def __init__(self, config, paras, task):
        super(Downstream_Solver, self).__init__(config, paras)

        # backup upstream settings        
        self.upstream_paras = copy.deepcopy(paras)
        self.upstream_config = copy.deepcopy(config)
        self.task = task # Downstream task the solver is solving
        
        # path and directories
        self.exp_name = self.exp_name.replace('transformer', task)
        self.logdir = self.paras.logdir.replace('transformer', task)
        self.ckpdir = self.ckpdir.replace('transformer', task)
        self.expdir = self.expdir.replace('transformer', task)
        self.dckpt = os.path.join(self.ckpdir, self.paras.dckpt)

        # model
        self.model_type = config['downstream']['model_type']
        self.load_model_list = config['downstream']['load_model_list']
        self.fine_tune = self.paras.fine_tune
        self.run_transformer = self.paras.run_transformer
        self.run_apc = self.paras.run_apc
        if self.fine_tune:  
            assert(self.run_transformer), 'Use `--run_transformer` to fine-tune the transformer model.'
            assert(not self.run_apc), 'Fine tuning only supports the transformer model.'
            assert(not self.paras.with_head), 'Fine tuning only supports the transformer model, not with head.'
        assert( not (self.run_transformer and self.run_apc) ), 'Transformer and Apc can not run at the same time!'
        if self.run_transformer and self.paras.with_head: self.verbose('Using transformer speech representations from head.')
        elif self.run_transformer and self.fine_tune: self.verbose('Fine-tuning on transformer speech representations.')
        elif self.run_transformer: self.verbose('Using transformer speech representations.')


    def load_data(self, split='train', load='cpc_phone'):
        ''' Load date for training / testing'''
        assert(load in ['montreal_phone', 'cpc_phone', 'sentiment', 'speaker', 'speaker_large']), 'Unsupported dataloader!'
        if load == 'montreal_phone' or load == 'cpc_phone' or load == 'speaker_large':
            if split == 'train':
                self.verbose('Loading source data from ' + str(self.config['dataloader']['train_set']) + ' from ' + self.config['dataloader']['data_path'])
                if load == 'montreal_phone' or load == 'cpc_phone': self.verbose('Loading phone data from ' + str(self.config['dataloader']['train_set']) + ' from ' + self.config['dataloader']['phone_path'])
            elif split == 'test': 
                if load != 'cpc_phone': self.verbose('Loading testing data ' + str(self.config['dataloader']['test_set']) + ' from ' + self.config['dataloader']['data_path'])
                if load == 'montreal_phone': self.verbose('Loading label data ' + str(self.config['dataloader']['test_set']) + ' from ' + self.config['dataloader']['phone_path'])
                elif load == 'cpc_phone': self.verbose('Loading label data from ' + self.config['dataloader']['phone_path'])
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
                run_mam=self.run_transformer, mam_config=self.transformer_config, \
                **self.config['dataloader']))


    def set_model(self, inference=False):
        input_dim = int(self.config['downstream'][self.model_type]['input_dim']) if \
                    self.config['downstream'][self.model_type]['input_dim'] != 'None' else None
        if 'transformer' in self.task:
            self.upstream_tester = Tester(self.upstream_config, self.upstream_paras)
            if self.fine_tune and inference: self.upstream_tester.load = False # During inference on fine-tuned model, load with `load_downstream_model()`
            self.upstream_tester.set_model(inference=True, with_head=self.paras.with_head) # inference should be set True so upstream solver won't create optimizer
            self.dr = self.upstream_tester.dr
            if input_dim is None:
                input_dim = self.transformer_config['hidden_size']
        elif 'apc' in self.task:
            self.apc = get_apc_model(path=self.paras.apc_path)
            if input_dim is None: 
                input_dim = self.transformer_config['hidden_size'] # use identical dim size for fair comparison
        elif 'baseline' in self.task:
            if input_dim is None: 
                if 'input_dim' in self.transformer_config:
                    input_dim = self.transformer_config['input_dim']
                else:
                    raise ValueError('Please update your config file to include the attribute `input_dim`.')
        else:
            raise NotImplementedError('Invalid Task!')

        if self.model_type == 'linear':
            self.classifier = LinearClassifier(input_dim=input_dim,
                                               class_num=self.dataloader.dataset.class_num,
                                               dconfig=self.config['downstream']['linear']).to(self.device)
        elif self.model_type == 'rnn':
            self.classifier = RnnClassifier(input_dim=input_dim,
                                            class_num=self.dataloader.dataset.class_num,
                                            dconfig=self.config['downstream']['rnn']).to(self.device)

        if not inference and self.fine_tune:
            # Setup Fine tune optimizer
            self.upstream_tester.transformer.train()
            param_optimizer = list(self.upstream_tester.transformer.named_parameters()) + list(self.classifier.named_parameters())
            self.optimizer = get_optimizer(params=param_optimizer,
                                           lr=self.learning_rate, 
                                           warmup_proportion=self.config['optimizer']['warmup_proportion'],
                                           training_steps=self.total_steps)
        elif not inference:
            self.optimizer = Adam(self.classifier.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
            self.classifier.train()
        else:
            self.classifier.eval()

        if self.load: # This will be set to True by default when Tester is running set_model()
            self.load_downstream_model(inference=inference)


    def save_model(self, name, model_all=True, assign_name=None):
        if model_all:
            all_states = {
                'Classifier': self.classifier.state_dict(),
                'Transformer': self.upstream_tester.transformer.state_dict() if self.fine_tune else None,
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
            model_path = f'{self.expdir}/{assign_name}.ckpt'
            torch.save(all_states, model_path)
            return

        new_model_path = '{}/{}-{}.ckpt'.format(self.expdir, name, self.global_step)
        torch.save(all_states, new_model_path)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)


    def load_downstream_model(self, inference=False):
        self.verbose('Load model from {}'.format(self.dckpt))
        all_states = torch.load(self.dckpt, map_location='cpu')
        
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
                self.verbose('@ Downstream, [Fine-Tuned Transformer] - Loading with Upstream Tester...')
                self.upstream_tester.load_model(inference=inference, from_path=self.ckpt)
                self.verbose('@ Downstream, [Fine-Tuned Transformer] - Loaded')
            except: self.verbose('[Fine-Tuned Transformer] - X')

        self.verbose('Model loading complete!')


###########
# TRAINER #
###########
class Downstream_Trainer(Downstream_Solver):
    ''' Handler for complete training progress'''
    def __init__(self, config, paras, task):
        super(Downstream_Trainer, self).__init__(config, paras, task)

        # Logger Settings
        self.logdir = os.path.join(self.logdir, self.exp_name)
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

        # mkdir
        if not os.path.exists(self.ckpdir): os.makedirs(self.ckpdir)
        if not os.path.exists(self.expdir): os.makedirs(self.expdir)


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
                    if self.run_transformer and self.paras.with_head:
                        # representations shape: (batch_size, seq_len, feature)
                        representations = self.upstream_tester.forward_with_head(features, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0))
                    elif self.run_transformer and self.fine_tune:
                        # representations shape: (batch_size, seq_len, feature)
                        representations = self.upstream_tester.forward_fine_tune(features, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0))
                    elif self.run_transformer:
                        # representations shape: (batch_size, layer, seq_len, feature)
                        representations = self.upstream_tester.forward(features, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0))
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
                        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.upstream_tester.transformer.parameters()) + list(self.classifier.parameters()), \
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
                        tmp_model_path = '{}/tmp.ckpt'.format(self.expdir)
                        new_dckpt = '/'.join(tmp_model_path.split('/')[-2:])
                        test_config = copy.deepcopy(self.upstream_config)
                        test_paras = copy.deepcopy(self.upstream_paras)
                        test_paras.dckpt = new_dckpt
                        tester = Downstream_Tester(test_config, test_paras, task=self.task)
                        tester.load_data(split=evaluation, load='cpc_phone' if 'cpc_phone' in self.task else self.task.split('_')[-1])
                        tester.set_model(inference=True)
                        eval_loss, eval_acc, eval_logits = tester.exec()
                        self.log.add_scalar(f'{evaluation}_loss', eval_loss, self.global_step)
                        self.log.add_scalar(f'{evaluation}_acc', eval_acc, self.global_step)
                        if eval_acc > best_val_acc:
                            self.verbose('Saving new best model on validation')
                            self.save_model(self.task, assign_name='best_val')
                            # torch.save(eval_logits, f'{self.exppdir}/best_val.logits') # uncomment to save logits
                            torch.cuda.empty_cache()
                            best_val_acc = eval_acc
                
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print('CUDA out of memory at step: ', self.global_step)
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                    else:
                        raise

                pbar.update(1)
                self.global_step += 1
                
        pbar.close()
        self.log.close()
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

                    if self.run_transformer and self.paras.with_head:
                        # representations shape: (batch_size, seq_len, feature)
                        representations = self.upstream_tester.forward_with_head(features, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0))
                    elif self.run_transformer and self.fine_tune:
                        # representations shape: (batch_size, seq_len, feature)
                        representations = self.upstream_tester.forward_fine_tune(features, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0))
                    elif self.run_transformer:
                        # representations shape: (batch_size, layer, seq_len, feature)
                        representations = self.upstream_tester.forward(features, process_from_loader=True)
                        features = self.up_sample_frames(features[0].squeeze(0))
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

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        if oom_counter >= 10: 
                            oom_counter = 0
                            break
                        else:
                            oom_counter += 1
                        print('CUDA out of memory during testing, aborting after ' + str(10 - oom_counter) + ' more tries...')
                        torch.cuda.empty_cache()
                    else:
                        raise

        average_loss = loss_sum / len(self.dataloader)
        test_acc = correct_count * 1.0 / valid_count
        self.verbose(f'Test result: loss {average_loss}, acc {test_acc}')

        return average_loss, test_acc, all_logits


def get_optimizer(params, lr, warmup_proportion, training_steps):
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