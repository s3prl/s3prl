# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ transformer/runner.py ]
#   Synopsis     [ runner for pre-training the transformer models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import time
import copy
import torch
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from tensorboardX import SummaryWriter
from transformer.model import TransformerConfig, TransformerForMaskedAcousticModel
from transformer.model_dual import DualTransformerConfig, DualTransformerForMaskedAcousticModel
from transformer.optimization import BertAdam, BertLamb, WarmupLinearSchedule
from transformer.mam import fast_position_encoding
from utility.audio import plot_spectrogram_to_numpy
from transformer.mam import process_train_MAM_data
from utility.preprocessor import OnlinePreprocessor
from small_models import SmallModelWrapper

LOG_WAV_NUM = 6


def logging(logger, step, tag, data, mode='scalar', preprocessor=None):
    if type(data) is torch.Tensor:
        data = data.detach().cpu()

    if mode == 'scalar':
        # data is a int or float
        logger.add_scalar(tag, data, global_step=step)
    elif mode == 'audio':
        # data: (seqlen, )
        assert preprocessor is not None
        data = data / data.abs().max().item()
        # log wavform
        logger.add_audio(f'{tag}.wav', data.reshape(-1, 1), global_step=step, sample_rate=preprocessor._sample_rate)
    else:
        raise NotImplementedError


##########
# RUNNER #
##########
class Runner():
    ''' Handler for complete pre-training progress of upstream models '''
    def __init__(self, args, config, dataloader, ckpdir):
        
        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.model_kept = []
        self.global_step = 1
        self.log = SummaryWriter(ckpdir)

        self.args = args
        self.config = config
        self.dataloader = dataloader
        self.ckpdir = ckpdir

        # optimizer
        self.learning_rate = float(config['optimizer']['learning_rate'])
        self.warmup_proportion = config['optimizer']['warmup_proportion']
        self.gradient_accumulation_steps = config['optimizer']['gradient_accumulation_steps']
        self.gradient_clipping = config['optimizer']['gradient_clipping']

        # Training details
        self.apex = config['runner']['apex']
        self.total_steps = config['runner']['total_steps']
        self.log_step = config['runner']['log_step']
        self.save_step = config['runner']['save_step']
        self.duo_feature = config['runner']['duo_feature']
        self.max_keep = config['runner']['max_keep']

        # model
        self.transformer_config = config['transformer']
        self.dr = config['transformer']['downsample_rate']
        self.dual_transformer = config['transformer']['dual_transformer'] if 'dual_transformer' in config['transformer'] else None
        if 'online' in config:
            print(f'[Runner] - Using features extracted on-the-fly')
            feat_list = [config['online']['input'], config['online']['target']]
            self.preprocessor = OnlinePreprocessor(**config['online'], feat_list=feat_list).to(device=self.device)
            self.input_dim, self.output_dim = [feat.size(-1) for feat in self.preprocessor()]
        else:
            print(f'[Runner] - Using features pre-extracted and saved')
            self.input_dim = self.transformer_config['input_dim']
            self.output_dim = 1025 if self.duo_feature else None # output dim is the same as input dim if not using duo features


    def set_model(self):
        print('[Runner] - Initializing Transformer model...')
        
        # build the Transformer model with speech prediction head
        if self.config.get('small_model') is not None:
            self.model = SmallModelWrapper(self.input_dim, self.output_dim, **self.config['small_model']).to(self.device)
        elif self.dual_transformer:
            model_config = DualTransformerConfig(self.config)
            self.model = DualTransformerForMaskedAcousticModel(model_config, self.input_dim, self.output_dim).to(self.device)
        else:
            model_config = TransformerConfig(self.config)
            self.model = TransformerForMaskedAcousticModel(model_config, self.input_dim, self.output_dim).to(self.device)
        self.model.train()

        if self.args.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[Runner] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[Runner] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # Setup optimizer
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if not hasattr(self.config['optimizer'], 'type'):
            self.config['optimizer']['type'] = 'adam'
        if self.apex:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                    lr=self.learning_rate,
                                    bias_correction=False,
                                    max_grad_norm=1.0)
            if self.config['optimizer']['loss_scale'] == 0:
                self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.config['optimizer']['loss_scale'])
            self.warmup_linear = WarmupLinearSchedule(warmup=self.warmup_proportion,
                                                      t_total=self.total_steps)
        elif self.config['optimizer']['type'] == 'adam':
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=self.learning_rate,
                                      warmup=self.warmup_proportion,
                                      t_total=self.total_steps,
                                      schedule='warmup_linear')
        elif self.config['optimizer']['type'] == 'lamb':
            self.optimizer = BertLamb(optimizer_grouped_parameters,
                                      lr=self.learning_rate,
                                      warmup=self.warmup_proportion,
                                      t_total=self.total_steps,
                                      schedule='warmup_linear')
        else:
            raise NotImplementedError()

        if self.args.resume is not None:
            self.load_model(self.args.resume)


    def load_model(self, ckptpth):
        ckpt = torch.load(ckptpth)

        if self.config.get('small_model') is not None:
            self.model.load_state_dict(ckpt['SmallModel'])
        else:
            self.model.Transformer.load_state_dict(ckpt['Transformer'])
            self.model.SpecHead.load_state_dict(ckpt['SpecHead'])
        self.optimizer.load_state_dict(ckpt['Optimizer'])
        self.global_step = ckpt['Global_step']


    def save_model(self, name='states', to_path=None):
        if self.config.get('small_model') is not None:
            all_states = {
                'SmallModel': self.model.state_dict(),
            }
        elif self.dual_transformer:
            all_states = {
                'SpecHead': self.model.SpecHead.state_dict() if not self.args.multi_gpu else self.model.module.SpecHead.state_dict(),
                'SpecTransformer': self.model.SpecTransformer.state_dict() if not self.args.multi_gpu else self.model.module.SpecTransformer.state_dict(),
                'SPE': self.model.SPE if not self.args.multi_gpu else self.model.module.SPE,
            }
            try: # store phonetic encoder if exist
                all_states['PhoneticTransformer'] = self.model.PhoneticTransformer.Transformer.state_dict() if not self.args.multi_gpu else self.model.module.PhoneticTransformer.Transformer.state_dict()
                all_states['PhoneticLayer'] = self.model.PhoneticTransformer.PhoneRecognizer.state_dict() if not self.args.multi_gpu else self.model.module.PhoneticTransformer.PhoneRecognizer.state_dict()
            except:
                pass
            try: # store speaker encoder if exist
                all_states['SpeakerTransformer'] = self.model.SpeakerTransformer.Transformer.state_dict() if not self.args.multi_gpu else self.model.module.SpeakerTransformer.Transformer.state_dict()
                all_states['SpeakerLayer'] = self.model.SpeakerTransformer.GlobalStyleToken.state_dict() if not self.args.multi_gpu else self.model.module.SpeakerTransformer.GlobalStyleToken.state_dict()
            except:
                pass
        else:
            all_states = {
                'SpecHead': self.model.SpecHead.state_dict() if not self.args.multi_gpu else self.model.module.SpecHead.state_dict(),
                'Transformer': self.model.Transformer.state_dict() if not self.args.multi_gpu else self.model.module.Transformer.state_dict(),
            }

        all_states['Optimizer'] = self.optimizer.state_dict()
        all_states['Global_step'] = self.global_step
        all_states['Settings'] = { 'Config': self.config, 'Paras': self.args }

        if to_path is None:
            new_model_path = '{}/{}-{}.ckpt'.format(self.ckpdir, name, self.global_step)
        else:
            new_model_path = to_path

        torch.save(all_states, new_model_path)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)


    def up_sample_frames(self, spec, return_first=False):
        if len(spec.shape) != 3: 
            spec = spec.unsqueeze(0)
            assert(len(spec.shape) == 3), 'Input should have acoustic feature of shape BxTxD'
        # spec shape: [batch_size, sequence_length // downsample_rate, output_dim * downsample_rate]
        spec_flatten = spec.view(spec.shape[0], spec.shape[1]*self.dr, spec.shape[2]//self.dr)
        if return_first: return spec_flatten[0]
        return spec_flatten # spec_flatten shape: [batch_size, sequence_length * downsample_rate, output_dim // downsample_rate]


    def down_sample_frames(self, spec):
        left_over = spec.shape[1] % self.dr
        if left_over != 0: spec = spec[:, :-left_over, :]
        spec_stacked = spec.view(spec.shape[0], spec.shape[1]//self.dr, spec.shape[2]*self.dr)
        return spec_stacked


    def process_data(self, spec):
        """Process training data for the masked acoustic model"""
        with torch.no_grad():
            
            assert(len(spec) == 5), 'dataloader should return (spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)'
            # Unpack and Hack bucket: Bucketing should cause acoustic feature to have shape 1xBxTxD'
            spec_masked = spec[0].squeeze(0)
            pos_enc = spec[1].squeeze(0)
            mask_label = spec[2].squeeze(0)
            attn_mask = spec[3].squeeze(0)
            spec_stacked = spec[4].squeeze(0)

            spec_masked = spec_masked.to(device=self.device)
            if pos_enc.dim() == 3:
                # pos_enc: (batch_size, seq_len, hidden_size)
                # GPU memory need (batch_size * seq_len * hidden_size)
                pos_enc = pos_enc.float().to(device=self.device)
            elif pos_enc.dim() == 2:
                # pos_enc: (seq_len, hidden_size)
                # GPU memory only need (seq_len * hidden_size) even after expanded
                pos_enc = pos_enc.float().to(device=self.device).expand(spec_masked.size(0), *pos_enc.size())
            mask_label = mask_label.bool().to(device=self.device)
            attn_mask = attn_mask.float().to(device=self.device)
            spec_stacked = spec_stacked.to(device=self.device)

        return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked # (x, pos_enc, mask_label, attention_mask. y)


    def process_dual_data(self, spec):
        """Process training data for the dual masked acoustic model"""
        with torch.no_grad():
            
            assert(len(spec) == 6), 'dataloader should return (time_masked, freq_masked, pos_enc, mask_label, attn_mask, spec_stacked)'
            # Unpack and Hack bucket: Bucketing should cause acoustic feature to have shape 1xBxTxD'
            time_masked = spec[0].squeeze(0)
            freq_masked = spec[1].squeeze(0)
            pos_enc = spec[2].squeeze(0)
            mask_label = spec[3].squeeze(0)
            attn_mask = spec[4].squeeze(0)
            spec_stacked = spec[5].squeeze(0)

            time_masked = time_masked.to(device=self.device)
            freq_masked = freq_masked.to(device=self.device)
            if pos_enc.dim() == 3:
                # pos_enc: (batch_size, seq_len, hidden_size)
                # GPU memory need (batch_size * seq_len * hidden_size)
                pos_enc = pos_enc.float().to(device=self.device)
            elif pos_enc.dim() == 2:
                # pos_enc: (seq_len, hidden_size)
                # GPU memory only need (seq_len * hidden_size) even after expanded
                pos_enc = pos_enc.float().to(device=self.device).expand(time_masked.size(0), *pos_enc.size())
            mask_label = mask_label.bool().to(device=self.device)
            attn_mask = attn_mask.float().to(device=self.device)
            spec_stacked = spec_stacked.to(device=self.device)

        return time_masked, freq_masked, pos_enc, mask_label, attn_mask, spec_stacked # (x, pos_enc, mask_label, attention_mask. y)


    def train(self):
        ''' Self-Supervised Pre-Training of Transformer Model'''

        pbar = tqdm(total=self.total_steps, dynamic_ncols=True)
        pbar.n = self.global_step - 1

        logging_temp = partial(logging, logger=self.log, preprocessor=copy.deepcopy(self.preprocessor).cpu())
        while self.global_step <= self.total_steps:

            progress = tqdm(self.dataloader, desc="Iteration", dynamic_ncols=True)

            step = 0
            loss_val = 0
            for batch in progress:
                if 'online' in self.config:
                    # batch are raw waveforms
                    # batch: (batch_size, channel, max_len)
                    wavs = batch.to(self.device)
                    specs = self.preprocessor(wavs)
                    batch = process_train_MAM_data(specs, config=self.transformer_config)
                    
                batch_is_valid, *batch = batch
                try:
                    if self.global_step > self.total_steps: break
                    if not batch_is_valid: continue
                    step += 1
                    
                    if self.config.get('small_model') is not None:
                        # spec_masked, spec_stacked should be both in linear-scale without log
                        spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = self.process_data(batch)
                        linears_inp = self.preprocessor(wavs, feat_list=[OnlinePreprocessor.get_feat_config('linear', 0)])[0]
                        loss, pred_spec = self.model(
                            feats_inp=spec_masked,
                            linears_inp=linears_inp,
                            linears_tar=spec_stacked,
                            mask_label=mask_label,
                        )
                        pred_spec = (F.relu(pred_spec) + 1e-12).log()
                        spec_stacked = (F.relu(spec_stacked) + 1e-12).log()
                    elif self.dual_transformer:
                        time_masked, freq_masked, pos_enc, mask_label, attn_mask, spec_stacked = self.process_dual_data(batch)
                        loss, pred_spec = self.model(time_masked, freq_masked, pos_enc, mask_label, attn_mask, spec_stacked)
                    else:
                        spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = self.process_data(batch)
                        loss, pred_spec = self.model(spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)
                    
                    # Accumulate Loss
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                    if self.apex and self.args.multi_gpu:
                        raise NotImplementedError
                    elif self.apex:
                        self.optimizer.backward(loss)
                    elif self.args.multi_gpu:
                        loss = loss.sum()
                        loss.backward()
                    else:
                        loss.backward()
                    loss_val += loss.item()

                    # Update
                    if (step+1) % self.gradient_accumulation_steps == 0:
                        if self.apex:
                            # modify learning rate with special warm up BERT uses
                            # if conifg.apex is False, BertAdam is used and handles this automatically
                            lr_this_step = self.learning_rate * self.warmup_linear.get_lr(self.global_step, self.warmup_proportion)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        
                        # Step
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                        if math.isnan(grad_norm):
                            print('[Runner] - Error : grad norm is NaN @ step ' + str(self.global_step))
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                        if self.global_step % self.log_step == 0:
                            # Log
                            self.log.add_scalar('lr', self.optimizer.get_lr()[0], self.global_step)
                            self.log.add_scalar('loss', (loss_val), self.global_step)
                            self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                            progress.set_description("Loss %.4f" % (loss_val))

                        if self.global_step % self.save_step == 0:
                            self.save_model('states')
                            
                            # tensorboard log
                            if self.dual_transformer: spec_masked = time_masked
                            spec_list = [spec_masked, pred_spec, spec_stacked]
                            name_list = ['mask_spec', 'pred_spec', 'true_spec']
                            if self.dual_transformer: 
                                spec_list.insert(1, freq_masked)
                                name_list.insert(1, 'mask_freq')
                                name_list[0] = 'mask_time'
                            
                            for i in range(len(spec_list)):
                                spec = self.up_sample_frames(spec_list[i][0], return_first=True)
                                spec = plot_spectrogram_to_numpy(spec.data.cpu().numpy())
                                self.log.add_image(name_list[i], spec, self.global_step)
                            
                            if self.config['transformer']['downsample_rate'] == 1 and 'online' in self.config:
                                phase_inp, phase_tar = self.preprocessor(wavs, feat_list=[
                                    OnlinePreprocessor.get_feat_config(feat_type='phase', channel=0),
                                    OnlinePreprocessor.get_feat_config(feat_type='phase', channel=1)
                                ])
                                wav_predicted = self.preprocessor.istft(pred_spec.exp(), phase_inp)
                                wav_target = self.preprocessor.istft(spec_stacked.exp(), phase_tar)
                                for i in range(LOG_WAV_NUM):
                                    logging_temp(step=self.global_step, tag=f'input-{i}', data=wavs[i, 0, :], mode='audio')
                                    logging_temp(step=self.global_step, tag=f'target-{i}', data=wav_target[i], mode='audio')
                                    logging_temp(step=self.global_step, tag=f'predicted-{i}', data=wav_predicted[i], mode='audio')

                            # if self.dual_transformer:
                            #     self.model.PhoneticTransformer.PhoneRecognizer.set_num_updates(self.global_step//1000)

                        loss_val = 0
                        pbar.update(1)
                        self.global_step += 1
                        sleep_time = self.config['runner'].get('sleep')
                        if sleep_time is not None:
                            time.sleep(sleep_time)
                        
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print('CUDA out of memory at step: ', self.global_step)
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                    else:
                        raise
                
        pbar.close()
        self.log.close()
