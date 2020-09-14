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
from collections import defaultdict
from tqdm import tqdm
from torch.optim import Adam
from tensorboardX import SummaryWriter
from downstream.solver import get_optimizer
from utility.mask_operations import *
from utility.preprocessor import OnlinePreprocessor
from downstream.model import *


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

        if args.disentangle:
            upstream_dim = upstream.config['transformer']['hidden_size']
            self.disentangler = eval(args.disentangler)(upstream_dim, **runner_config[args.disentangler]
                                                        if args.disentangler in runner_config else {}).to(self.device)
            _, no_speaker, _ = self.disentangler(torch.randn(1, 100, upstream_dim).to(self.device))
            self.classifier = eval(args.classifier)(no_speaker.size(-1), dataloader['train'].dataset.class_num, **runner_config[args.classifier]
                                                    if args.classifier in runner_config else {}).to(self.device)

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

        if self.args.disentangle:
            self.disentangler_optimizer = Adam(self.disentangler.parameters(), lr=float(self.config['learning_rate']), betas=(0.9, 0.999))
            self.classifier_optimizer = Adam(self.classifier.parameters(), lr=float(self.config['learning_rate']), betas=(0.9, 0.999))
            self.upstream_model.eval()
            self.downstream_model.eval()
            self.disentangler.train()
            self.classifier.train()

        if self.args.resume is not None:
            self.load_model(self.args.resume)


    def load_model(self, ckptpth):
        ckpt = torch.load(ckptpth)
        if self.args.fine_tune:
            self.upstream_model.load_state_dict(ckpt['Upstream'])
        self.downstream_model.load_state_dict(ckpt['Downstream'])
        self.optimizer.load_state_dict(ckpt['Optimizer'])
        self.global_step = ckpt['Global_step']


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
        pbar.n = self.global_step - 1

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

                    if self.args.cmvn:
                        features = mask_normalize(features, label_mask.unsqueeze(-1))

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

                    if self.args.cmvn:
                        features = mask_normalize(features, label_mask.unsqueeze(-1))

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


    def get_speaker_embeddings(self):
        self.upstream_model.eval()
        self.downstream_model.eval()

        all_embeddings = defaultdict(list)
        for features, labels in tqdm(self.dataloader[self.args.inference_split], desc="Iteration"):
            with torch.no_grad():
                # features: (1, batch_size, seq_len, feature)
                # dimension of labels depend on task and dataset, but the first dimention is always trivial due to bucketing, eg. (1, ...)
                wav_inp = features.squeeze(0).to(device=self.device, dtype=torch.float32)
                phase_inp = self.upstream_model.preprocessor(wav_inp.transpose(1, 2), feat_list=[
                    OnlinePreprocessor.get_feat_config('phase', 0)
                ])[0]

                features = self.upstream_model(wav_inp)

                # Since zero padding technique, some timestamps of features are not valid
                # For each timestamps, we mark 1 on valid timestamps, and 0 otherwise
                # This variable can be useful for frame-wise metric, like phoneme recognition or speaker verification
                # label_mask: (batch_size, seq_len), LongTensor
                labels = labels.squeeze(0).to(device=self.device)
                label_mask = (features.sum(dim=-1) != 0).type(torch.LongTensor).to(device=self.device, dtype=torch.long)
                valid_lengths = label_mask.sum(dim=1)

                embeddings = mask_mean(features, label_mask.unsqueeze(-1))
                for embedding, label in zip(embeddings, labels):
                    all_embeddings[label.item()].append(embedding.detach().cpu())
        
        mean_embeddings = {}
        for key, value in all_embeddings.items():
            mean_embeddings[key] = torch.stack(value, dim=0).mean(dim=0)
        
        return mean_embeddings


    def generate(self):
        ''' Testing of downstream tasks'''

        self.upstream_model.eval()
        self.downstream_model.eval()

        speaker_embeddings = self.get_speaker_embeddings()

        dataset = self.dataloader[self.args.inference_split].dataset
        sample_indices = range(0, dataset.__len__(), dataset.__len__() // self.args.sample_num)
        for indice in tqdm(list(sample_indices), desc="Iteration"):
            features, labels = dataset[indice]
            with torch.no_grad():
                # features: (1, batch_size, seq_len, feature)
                # dimension of labels depend on task and dataset, but the first dimention is always trivial due to bucketing, eg. (1, ...)
                wav_inp = features.squeeze(0).to(device=self.device, dtype=torch.float32)
                phase_inp = self.upstream_model.preprocessor(wav_inp.transpose(1, 2), feat_list=[
                    OnlinePreprocessor.get_feat_config('phase', 0)
                ])[0]

                features = self.upstream_model(wav_inp)

                # Since zero padding technique, some timestamps of features are not valid
                # For each timestamps, we mark 1 on valid timestamps, and 0 otherwise
                # This variable can be useful for frame-wise metric, like phoneme recognition or speaker verification
                # label_mask: (batch_size, seq_len), LongTensor
                labels = labels.squeeze(0).to(device=self.device)
                label_mask = (features.sum(dim=-1) != 0).type(torch.LongTensor).to(device=self.device, dtype=torch.long)
                valid_lengths = label_mask.sum(dim=1)

                means = mask_mean(features, label_mask.unsqueeze(-1))
                if self.args.assign_speaker > -1:
                    sampled_speakerid = self.args.assign_speaker
                else:
                    sampled_speakerid = random.randint(0, len(speaker_embeddings.keys()) - 1)
                
                converted = (features - means) + speaker_embeddings[sampled_speakerid].to(features.device)
                linear_out = self.downstream_model(converted)
                wav_out = self.upstream_model.preprocessor.istft(linear_out, phase_inp)
            
                for idx, (wi, wo, l) in enumerate(zip(wav_inp, wav_out, labels)):
                    self.log.add_audio(f'{indice}-{idx}-source-speaker{dataset.idx2speaker[l.item()]}.wav', wi.reshape(-1, 1), global_step=self.global_step,
                                        sample_rate=self.upstream_model.preprocessor._sample_rate)
                    self.log.add_audio(f'{indice}-{idx}-target-speaker{dataset.idx2speaker[sampled_speakerid]}.wav', wo.reshape(-1, 1), global_step=self.global_step,
                                        sample_rate=self.upstream_model.preprocessor._sample_rate)


        self.log.close()


    def disentangle(self):
        ''' Training of downstream tasks'''

        pbar = tqdm(total=int(self.config['total_steps']))
        pbar.n = self.global_step - 1

        d_loses = 0.0
        c_loses = 0.0

        while self.global_step <= int(self.config['total_steps']):

            for features, labels in tqdm(self.dataloader['train'], desc="Iteration"):
                try:
                    if self.global_step > int(self.config['total_steps']): break
                    
                    # features: (1, batch_size, seq_len, feature)
                    # dimension of labels depend on task and dataset, but the first dimention is always trivial due to bucketing, eg. (1, ...)
                    features = features.squeeze(0).to(device=self.device, dtype=torch.float32)
                    real_spec = self.upstream_model.preprocessor(features.transpose(1, 2).contiguous())[1]
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

                    no_speaker, speaker, reconstructed_representation = self.disentangler(features)
                    reconstructed_spec = self.downstream_model(reconstructed_representation)
                    d_loss = F.l1_loss((reconstructed_spec + 1e-8).log(), (real_spec + 1e-8).log())
                    speaker_prediction = self.classifier(no_speaker)
                    c_loss = F.cross_entropy(speaker_prediction, labels)

                    # record
                    c_loses += c_loss.item() / self.config['classifier_iteration']
                    d_loses += d_loss.item() / self.config['disentangler_iteration']

                    if self.global_step % self.config['classifier_iteration'] == 0:
                        c_loss.backward()
                        self.classifier_optimizer.step()
                        self.classifier_optimizer.zero_grad()

                    if self.global_step % self.config['disentangler_iteration'] == 0:
                        d_loss.backward()
                        self.disentangler_optimizier.step()
                        self.disentangler_optimizier.zero_grad()
                    
                    # logging
                    if self.global_step % int(self.config['log_step']) == 0:
                        # Log
                        c_los = c_loses / int(self.config['log_step'])
                        d_los = d_loses / int(self.config['log_step'])
                        self.log.add_scalar('classifier loss', c_los, self.global_step)
                        self.log.add_scalar('disentangler loss', d_los, self.global_step)
                        pbar.set_description('c_los %.5f, d_los %.5f' % (c_los, d_los))
                        c_loses = 0.0
                        d_loses = 0.0

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