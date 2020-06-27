# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ downstream/runner.py ]
#   Synopsis     [ runner to perform downstream train/dev/test ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import random
from tqdm import tqdm
from torch.optim import Adam
from tensorboardX import SummaryWriter
from downstream.solver import get_optimizer


##########
# RUNNER #
##########
class Runner():
    ''' Handler for complete training and evaluation progress of downstream models '''
    def __init__(self, args, runner_config, dataloader, upstream, downstream, expdir):

        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.model_kept = []
        self.global_step = 1
        self.log = SummaryWriter(expdir)

        self.args = args
        self.config = runner_config
        self.dataloader = dataloader
        self.upstream_model = upstream.to(self.device)
        self.downstream_model = downstream.to(self.device)
        self.expdir = expdir


    def set_model(self):

        if self.args.fine_tune:
            # Setup Fine tune optimizer
            self.upstream_model.train()
            param_optimizer = list(self.upstream_model.named_parameters()) + list(self.downstream_model.named_parameters())
            self.optimizer = get_optimizer(params=param_optimizer,
                                           lr=float(self.config['learning_rate']), 
                                           warmup_proportion=float(self.config['warmup_proportion']),
                                           training_steps=int(self.config['total_steps']))
        else:
            self.upstream_model.eval()
            self.optimizer = Adam(self.downstream_model.parameters(), lr=float(self.config['learning_rate']), betas=(0.9, 0.999))
        
        self.downstream_model.train()


    def save_model(self, name='states', save_best=None):
        
        all_states = {
            'Upstream': self.upstream_model.state_dict() if self.args.fine_tune else None,
            'Downstream': self.downstream_model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Global_step': self.global_step,
            'Settings': {
                'Config': self.config,
                'Paras': self.args,
            },
        }

        if save_best is not None:
            model_path = f'{self.expdir}/{save_best}.ckpt'
            torch.save(all_states, model_path)
            return

        model_path = f'{self.expdir}/{name}-{self.global_step}.ckpt'
        torch.save(all_states, model_path)
        self.model_kept.append(model_path)

        if len(self.model_kept) >= int(self.config['max_keep']):
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)


    def train(self):
        ''' Training of downstream tasks'''

        pbar = tqdm(total=int(self.config['total_steps']))
        corrects = 0
        valids = 0
        best_acc = 0.0
        best_eval_acc = 0.0
        best_test_acc = 0.0
        loses = 0.0

        while self.global_step <= int(self.config['total_steps']):

            for features, labels in tqdm(self.dataloader['train'], desc="Iteration"):
                try:
                    if self.global_step > int(self.config['total_steps']): break
                    
                    # features: (1, batch_size, seq_len, feature)
                    # dimension of labels depend on task and dataset, but the first dimention is always trivial due to bucketing, eg. (1, ...)
                    features = features.squeeze(0).to(device=self.device, dtype=torch.float32)
                    if self.args.fine_tune:
                        features = self.upstream_model(features)
                    else:
                        with torch.no_grad():
                            features = self.upstream_model(features)

                    # Since zero padding technique, some timestamps of features are not valid
                    # For each timestamps, we mark 1 on valid timestamps, and 0 otherwise
                    # This variable can be useful for frame-wise metric, like phoneme recognition or speaker verification
                    # label_mask: (batch_size, seq_len), LongTensor
                    # valid_lengths: (batch_size), LongTensor
                    labels = labels.squeeze(0).to(device=self.device)  # labels can be torch.long or torch.float (regression)
                    label_mask = (features.sum(dim=-1) != 0).type(torch.LongTensor).to(device=self.device, dtype=torch.long)
                    valid_lengths = label_mask.sum(dim=1)

                    if 'utterance' in self.args.run:
                        # labels: (batch_size, )
                        loss, _, correct, valid = self.downstream_model(features, labels, valid_lengths)
                    else:
                        # labels: (batch_size, seq_len)
                        loss, _, correct, valid = self.downstream_model(features, labels, label_mask)

                    # Accumulate Loss
                    loss.backward()

                    # record
                    loses += loss.detach().item()
                    corrects += correct
                    valids += valid

                    # gradient clipping
                    if self.args.fine_tune: 
                        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.upstream_model.parameters()) + list(self.downstream_model.parameters()), \
                                                                   float(self.config['gradient_clipping']))
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.downstream_model.parameters(), \
                                                                   float(self.config['gradient_clipping']))
                    
                    # step
                    if math.isnan(grad_norm):
                        print('[Runner] - Error : grad norm is NaN @ step ' + str(self.global_step))
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    # logging
                    if self.global_step % int(self.config['log_step']) == 0:
                        # Log
                        acc = corrects.item() / valids.item()
                        los = loses / int(self.config['log_step'])
                        self.log.add_scalar('acc', acc, self.global_step)
                        self.log.add_scalar('loss', los, self.global_step)
                        self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                        pbar.set_description('Loss %.5f, Acc %.5f' % (los, acc))

                        loses = 0.0
                        corrects = 0
                        valids = 0

                    if self.global_step % int(self.config['save_step']) == 0 and acc > best_acc:
                        self.save_model()
                        best_acc = acc

                    # evaluate on the self.config['evaluation'] set
                    if self.global_step % int(self.config['dev_step']) == 0:
                        print('[Runner] - Evaluating on: ', self.config['evaluation'])
                        torch.cuda.empty_cache()
                        eval_loss, eval_acc, _ = self.evaluate(split=self.config['evaluation'])
                        self.log.add_scalar(f"{self.config['evaluation']}_loss", eval_loss, self.global_step)
                        self.log.add_scalar(f"{self.config['evaluation']}_acc", eval_acc, self.global_step)
                        if eval_acc > best_eval_acc:
                            print('[Runner] - Saving new best model on: ', self.config['evaluation'])
                            self.save_model(save_best=f"best_{self.config['evaluation']}")
                            torch.cuda.empty_cache()
                            best_eval_acc = eval_acc
                        
                        # evaluate on the test set if not already
                        if self.config['evaluation'] != 'test':
                            torch.cuda.empty_cache()
                            test_loss, test_acc, _ = self.evaluate(split='test')
                            self.log.add_scalar('test_loss', test_loss, self.global_step)
                            self.log.add_scalar('test_acc', test_acc, self.global_step)
                            if test_acc > best_test_acc:
                                print('[Runner] - Saving new best model on: ', 'test')
                                self.save_model(save_best='best_test')
                                torch.cuda.empty_cache()
                                best_test_acc = test_acc

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print('[Runner] - CUDA out of memory at step: ', self.global_step)
                    elif 'worker' in str(e):
                        print('[Runner] - Dataloader worker not available at step: ', self.global_step)
                    else:
                        raise
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()

                pbar.update(1)
                self.global_step += 1
                
        pbar.close()
        self.log.close()


    def evaluate(self, split):
        ''' Testing of downstream tasks'''

        self.upstream_model.eval()
        self.downstream_model.eval()
        
        valid_count = 0
        correct_count = 0
        loss_sum = 0
        all_logits = []

        oom_counter = 0
        for features, labels in tqdm(self.dataloader[split], desc="Iteration"):
            with torch.no_grad():
                try:
                    # features: (1, batch_size, seq_len, feature)
                    # dimension of labels depend on task and dataset, but the first dimention is always trivial due to bucketing, eg. (1, ...)
                    features = features.squeeze(0).to(device=self.device, dtype=torch.float32)
                    features = self.upstream_model(features)

                    # Since zero padding technique, some timestamps of features are not valid
                    # For each timestamps, we mark 1 on valid timestamps, and 0 otherwise
                    # This variable can be useful for frame-wise metric, like phoneme recognition or speaker verification
                    # label_mask: (batch_size, seq_len), LongTensor
                    labels = labels.squeeze(0).to(device=self.device)
                    label_mask = (features.sum(dim=-1) != 0).type(torch.LongTensor).to(device=self.device, dtype=torch.long)
                    valid_lengths = label_mask.sum(dim=1)

                    if 'utterance' in self.args.run:
                        # labels: (batch_size, )
                        loss, logits, correct, valid = self.downstream_model(features, labels, valid_lengths)
                    else:
                        # labels: (batch_size, seq_len)
                        loss, logits, correct, valid = self.downstream_model(features, labels, label_mask)
                    
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
                        print(f'[Runner] - CUDA out of memory during {split}ing, aborting after ' + str(10 - oom_counter) + ' more tries...')
                        torch.cuda.empty_cache()
                    else:
                        raise

        average_loss = loss_sum / len(self.dataloader[split])
        eval_acc = correct_count * 1.0 / valid_count
        print(f'[Runner] - {split} result: loss {average_loss}, acc {eval_acc}')
        
        self.upstream_model.train()
        self.downstream_model.train()

        return average_loss, eval_acc, all_logits