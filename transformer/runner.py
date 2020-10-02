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
import torch
import random
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformer.model import TransformerConfig, TransformerForMaskedAcousticModel
from transformer.model_dual import DualTransformerConfig, DualTransformerForMaskedAcousticModel
from transformer.optimization import BertAdam, Lamb, WarmupLinearSchedule
from transformer.mam import fast_position_encoding
from utility.audio import plot_spectrogram_to_numpy


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
        self.dual_transformer = config['transformer']['dual_transformer'] if 'dual_transformer' in config['transformer'] else False
        self.wave_transformer = config['transformer']['wave_transformer'] if 'wave_transformer' in config['transformer'] else False
        if 'online' in config:
            print(f'[Runner] - Using features extracted on-the-fly')
            self.input_dim, self.output_dim = [feat.size(-1) for feat in self.dataloader.dataset.preprocessor()]
        else:
            if self.wave_transformer: raise ValueError('Wave transformer needs to be run with online feature extraction!')
            print(f'[Runner] - Using features pre-extracted and saved')
            self.input_dim = self.transformer_config['input_dim']
            self.output_dim = 1025 if self.duo_feature else None # output dim is the same as input dim if not using duo features


    def set_model(self):
        # build the Transformer model with speech prediction head
        if self.dual_transformer:
            print('[Runner] - Initializing Dual Transformer model...')
            model_config = DualTransformerConfig(self.config)
            self.model = DualTransformerForMaskedAcousticModel(model_config, self.input_dim, self.output_dim).to(self.device)
        else:
            if self.wave_transformer:
                print('[Runner] - Initializing Wave Transformer model...')
            else:
                print('[Runner] - Initializing Transformer model...')
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

        if 'type' not in self.config['optimizer']:
            self.config['optimizer']['type'] = 'adam'
        print('[Runner] - Optimizer: ' + ('apex Fused Adam' if self.apex else str(self.config['optimizer']['type'])))
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
        elif self.config['optimizer']['type'] == 'lamb' or self.config['optimizer']['type'] == 'adamW':
            self.optimizer = Lamb(optimizer_grouped_parameters,
                                      lr=self.learning_rate,
                                      warmup=self.warmup_proportion,
                                      t_total=self.total_steps,
                                      schedule='warmup_linear',
                                      adam=True if self.config['optimizer']['type'] == 'adamW' else False,
                                      correct_bias=True if self.config['optimizer']['type'] == 'adamW' else False)
        else:
            raise NotImplementedError()


    def save_model(self, name='states', to_path=None):
        if self.dual_transformer:
            all_states = { 'SpecHead': self.model.SpecHead.state_dict() if not self.args.multi_gpu else self.model.module.SpecHead.state_dict() }
            if hasattr(self.model, 'SpecTransformer'):
                all_states['SpecTransformer'] = self.model.SpecTransformer.state_dict() if not self.args.multi_gpu else self.model.module.SpecTransformer.state_dict()
            if hasattr(self.model, 'SPE'):
                all_states['SPE'] = self.model.SPE if not self.args.multi_gpu else self.model.module.SPE
            if hasattr(self.model, 'PhoneticTransformer'): 
                all_states['PhoneticTransformer'] = self.model.PhoneticTransformer.Transformer.state_dict() if not self.args.multi_gpu else self.model.module.PhoneticTransformer.Transformer.state_dict()
            if hasattr(self.model.PhoneticTransformer, 'PhoneRecognizer'): 
                all_states['PhoneticLayer'] = self.model.PhoneticTransformer.PhoneRecognizer.state_dict() if not self.args.multi_gpu else self.model.module.PhoneticTransformer.PhoneRecognizer.state_dict()
            if hasattr(self.model, 'SpeakerTransformer'): 
                all_states['SpeakerTransformer'] = self.model.SpeakerTransformer.Transformer.state_dict() if not self.args.multi_gpu else self.model.module.SpeakerTransformer.Transformer.state_dict()
            if hasattr(self.model.SpeakerTransformer, 'SpeakerRecognizer'): 
                all_states['SpeakerLayer'] = self.model.SpeakerTransformer.SpeakerRecognizer.state_dict() if not self.args.multi_gpu else self.model.module.SpeakerTransformer.SpeakerRecognizer.state_dict()
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

        pbar = tqdm(total=self.total_steps)
        while self.global_step <= self.total_steps:

            progress = tqdm(self.dataloader, desc="Iteration")

            step = 0
            loss_val = 0
            for batch in progress:
                batch_is_valid, *batch = batch
                try:
                    if self.global_step > self.total_steps: break
                    if not batch_is_valid: continue
                    step += 1
                    
                    if self.dual_transformer:
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
                                if i == 0 and self.wave_transformer:
                                    self.log.add_audio(name_list[0], spec_list[0][0].data.cpu().numpy(), self.global_step, self.config['online']['sample_rate'])
                                    continue
                                spec = self.up_sample_frames(spec_list[i][0], return_first=True)
                                spec = plot_spectrogram_to_numpy(spec.data.cpu().numpy())
                                self.log.add_image(name_list[i], spec, self.global_step)

                        loss_val = 0
                        pbar.update(1)
                        self.global_step += 1
                        
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print('CUDA out of memory at step: ', self.global_step)
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                    else:
                        raise
                
        pbar.close()
        self.log.close()
