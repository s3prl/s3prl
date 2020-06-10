# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ transformer/solver.py ]
#   Synopsis     [ solvers for the transformer model: trainer / tester ]
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
from shutil import copyfile
from tqdm import tqdm, trange
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataloader import get_Dataloader
from transformer.model import TransformerConfig, TransformerModel, TransformerForMaskedAcousticModel
from transformer.optimization import BertAdam, WarmupLinearSchedule
from transformer.mam import fast_position_encoding
from utility.audio import plot_spectrogram_to_numpy, plot_spectrogram, plot_embedding, plot_attention
from utility.audio import sample_rate, inv_spectrogram


##########
# SOLVER #
##########
class Solver():
    ''' Super class Solver for all kinds of tasks'''
    def __init__(self, config, paras):
        
        # General Settings
        self.config = config
        self.paras = paras
        self.transformer_config = config['transformer']
        self.device = torch.device('cuda') if (self.paras.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): self.verbose('CUDA is available!')

        # path and directories
        self.exp_name = paras.name
        if self.exp_name is None:
            self.exp_name = '_'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
        self.ckpdir = paras.ckpdir
        self.expdir = os.path.join(self.ckpdir, self.exp_name)
        
        self.load = paras.load
        # only for test
        self.ckpt = os.path.join(self.ckpdir, paras.ckpt)

        # model
        self.load_model_list = config['solver']['load_model_list']
        self.duo_feature = config['solver']['duo_feature']
        self.output_dim = 1025 if self.duo_feature else None # output dim is the same as input dim if not using duo features
        if 'input_dim' in self.transformer_config:
            self.input_dim = self.transformer_config['input_dim']
        else:
            raise ValueError('Please update your config file to include the attribute `input_dim`.')


    def verbose(self, msg, end='\n'):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            print('[SOLVER] - ', msg, end=end)


    def load_data(self, split='train'):
        ''' Load data for training / testing'''
        if split == 'train': 
            self.verbose('Loading source data ' + str(self.config['dataloader']['train_set']) + ' from ' + self.config['dataloader']['data_path'])
            if self.duo_feature: self.verbose('Loading target data ' + str(self.config['dataloader']['train_set']) + ' from ' + self.config['dataloader']['target_path'])
        elif split == 'test': 
            self.verbose('Loading testing data ' + str(self.config['dataloader']['test_set']) + ' from ' + self.config['dataloader']['data_path'])
        else:
            raise NotImplementedError('Invalid `split` argument!')

        if self.duo_feature:
            setattr(self, 'dataloader', get_Dataloader(split, load='duo', use_gpu=self.paras.gpu, \
                    mam_config=self.transformer_config, **self.config['dataloader'])) # run_mam is automatically performed
        else:
            setattr(self, 'dataloader', get_Dataloader(split, load='acoustic', use_gpu=self.paras.gpu, run_mam=True, \
                    mam_config=self.transformer_config, **self.config['dataloader']))


    def set_model(self, inference=False, with_head=False, from_path=None, output_attention=False):
        self.verbose('Initializing Transformer model.')
        
        # uild the Transformer model with speech prediction head
        self.model_config = TransformerConfig(self.config)
        self.dr = self.model_config.downsample_rate
        self.hidden_size = self.model_config.hidden_size
        self.with_head = with_head
        self.output_attention = output_attention
        
        if not inference or with_head:
            self.model = TransformerForMaskedAcousticModel(self.model_config, self.input_dim, self.output_dim, self.output_attention).to(self.device)
            self.transformer = self.model.Transformer
            if self.paras.multi_gpu:
                self.model = torch.nn.DataParallel(self.model)
                self.transformer = torch.nn.DataParallel(self.transformer)
                self.verbose('Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
            self.verbose('Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        if inference and not with_head:
            self.transformer = TransformerModel(self.model_config, self.input_dim, self.output_attention).to(self.device)
            if self.paras.multi_gpu:
                self.transformer = torch.nn.DataParallel(self.transformer)
                self.verbose('Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
            self.verbose('Number of parameters: ' + str(sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)))
            self.transformer.eval()
        elif inference and with_head:
            self.model.eval()
        elif not inference:
            self.model.train()

            # Setup optimizer
            param_optimizer = list(self.model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

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
            else:
                self.optimizer = BertAdam(optimizer_grouped_parameters,
                                        lr=self.learning_rate,
                                        warmup=self.warmup_proportion,
                                        t_total=self.total_steps)
        else:
            raise NotImplementedError('Invalid Arguments!')

        if self.load: # This will be set to True by default when Tester is running set_model()
            self.load_model(inference=inference, with_head=with_head, from_path=from_path)


    def save_model(self, name='states', model_all=True, to_path=None):
        if model_all:
            all_states = {
                'SpecHead': self.model.SpecHead.state_dict() if not self.paras.multi_gpu else self.model.module.SpecHead.state_dict(),
                'Transformer': self.transformer.state_dict() if not self.paras.multi_gpu else self.transformer.module.state_dict(),
                'Optimizer': self.optimizer.state_dict(),
                'Global_step': self.global_step,
                'Settings': {
                    'Config': self.config,
                    'Paras': self.paras,
                },
            }
        else:
            all_states = {
                'Transformer': self.transformer.state_dict() if not self.paras.multi_gpu else self.transformer.module.state_dict(),
                'Settings': {
                    'Config': self.config,
                    'Paras': self.paras,
                },
            }
        if to_path is None:
            new_model_path = '{}/{}-{}.ckpt'.format(self.expdir, name, self.global_step)
        else: new_model_path = to_path
        torch.save(all_states, new_model_path)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)


    def load_model(self, inference=False, with_head=False, from_path=None):
        if from_path is not None:
            self.verbose('Load model from {}'.format(from_path))
            all_states = torch.load(from_path, map_location='cpu')
            self.load_model_list = ['Transformer']
        else:
            self.verbose('Load model from {}'.format(self.ckpt))
            all_states = torch.load(self.ckpt, map_location='cpu')

        if 'SpecHead' in self.load_model_list:
            if not inference or with_head:
                try:
                    if not self.paras.multi_gpu:
                        self.model.SpecHead.load_state_dict(all_states['SpecHead'])
                    else:
                        self.model.module.SpecHead.load_state_dict(all_states['SpecHead'])
                    self.verbose('[SpecHead] - Loaded')
                except: self.verbose('[SpecHead - X]')
                
        if 'Transformer' in self.load_model_list:
            try:
                state_dict = all_states['Transformer'] 

                # Load from a PyTorch state_dict
                old_keys = []
                new_keys = []
                for key in state_dict.keys():
                    new_key = None
                    if 'gamma' in key:
                        new_key = key.replace('gamma', 'weight')
                    if 'beta' in key:
                        new_key = key.replace('beta', 'bias')
                    if new_key:
                        old_keys.append(key)
                        new_keys.append(new_key)
                for old_key, new_key in zip(old_keys, new_keys):
                    state_dict[new_key] = state_dict.pop(old_key)

                missing_keys = []
                unexpected_keys = []
                error_msgs = []
                # copy state_dict so _load_from_state_dict can modify it
                metadata = getattr(state_dict, '_metadata', None)
                state_dict = state_dict.copy()
                if metadata is not None:
                    state_dict._metadata = metadata

                def load(module, prefix=''):
                    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                    module._load_from_state_dict(
                        state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                    for name, child in module._modules.items():
                        if child is not None:
                            load(child, prefix + name + '.')

                # perform load
                if not self.paras.multi_gpu:
                    load(self.transformer)
                else:
                    load(self.transformer.module)

                if len(missing_keys) > 0:
                    self.verbose("Weights of {} not initialized from pretrained model: {}".format(
                        self.transformer.__class__.__name__, missing_keys))
                if len(unexpected_keys) > 0:
                    self.verbose("Weights from pretrained model not used in {}: {}".format(
                        self.transformer.__class__.__name__, unexpected_keys))
                if len(error_msgs) > 0:
                    raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                       self.transformer.__class__.__name__, "\n\t".join(error_msgs)))
                self.verbose('[Transformer] - Loaded')
            except: self.verbose('[Transformer - X]')

        if 'Optimizer' in self.load_model_list and not inference:
            try:
                self.optimizer.load_state_dict(all_states['Optimizer'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                self.verbose('[Optimizer] - Loaded')
            except: self.verbose('[Optimizer - X]')

        if 'Global_step' in self.load_model_list and not inference:
            try:
                self.global_step = all_states['Global_step']
                self.verbose('[Global_step] - Loaded')
            except: self.verbose('[Global_step - X]')

        self.verbose('Model loading complete!')


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


    def position_encoding(self, seq_len, batch_size=None, padding_idx=None):
        ''' Sinusoid position encoding table '''
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / self.hidden_size)
     
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(self.hidden_size)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(seq_len)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            sinusoid_table[padding_idx:] = 0. # zero vector for padding dimension

        if batch_size is not None:
            batch_sinusoid_table = np.repeat(sinusoid_table[np.newaxis,...], batch_size, axis=0)
            return batch_sinusoid_table # (batch_size, seq_len, hidden_size)
        else:
            return sinusoid_table  # (seq_len, hidden_size)


###########
# TRAINER #
###########
class Trainer(Solver):
    ''' Handler for complete training progress'''
    def __init__(self, config, paras):
        super(Trainer, self).__init__(config, paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir, self.exp_name)
        self.log = SummaryWriter(self.logdir)

        # Training details
        self.apex = config['solver']['apex']
        self.log_step = config['solver']['log_step']
        self.save_step = config['solver']['save_step']
        self.total_steps = config['solver']['total_steps']
        self.learning_rate = float(self.config['optimizer']['learning_rate'])
        self.warmup_proportion = self.config['optimizer']['warmup_proportion']
        self.gradient_accumulation_steps = self.config['optimizer']['gradient_accumulation_steps']
        self.gradient_clipping = self.config['optimizer']['gradient_clipping']
        self.max_keep = config['solver']['max_keep']
        self.reset_train()

        # mkdir
        if not os.path.exists(self.ckpdir): os.makedirs(self.ckpdir)
        if not os.path.exists(self.expdir): os.makedirs(self.expdir)
        copyfile(self.paras.config, os.path.join(self.expdir, self.paras.config.split('/')[-1]))

    def reset_train(self):
        self.model_kept = []
        self.global_step = 0


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
                pos_enc = torch.FloatTensor(pos_enc).to(device=self.device)
            elif pos_enc.dim() == 2:
                # pos_enc: (seq_len, hidden_size)
                # GPU memory only need (seq_len * hidden_size) even after expanded
                pos_enc = torch.FloatTensor(pos_enc).to(device=self.device).expand(spec_masked.size(0), *pos_enc.size())
            mask_label = torch.ByteTensor(mask_label).to(device=self.device)
            attn_mask = torch.FloatTensor(attn_mask).to(device=self.device)
            spec_stacked = spec_stacked.to(device=self.device)

        return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked # (x, pos_enc, mask_label, attention_mask. y)


    def exec(self):
        ''' Self-Supervised Pre-Training of Transformer Model'''
        self.verbose('Training set total ' + str(len(self.dataloader)) + ' batches.')

        pbar = tqdm(total=self.total_steps)
        while self.global_step <= self.total_steps:

            progress = tqdm(self.dataloader, desc="Iteration")

            step = 0
            for batch_is_valid, *batch in progress:
                try:
                    if self.global_step > self.total_steps: break
                    if not batch_is_valid: continue
                    step += 1
                    
                    spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = self.process_data(batch)
                    loss, pred_spec = self.model(spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)
                    
                    # Accumulate Loss
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                    if self.apex and self.paras.multi_gpu:
                        raise NotImplementedError
                    elif self.apex:
                        self.optimizer.backward(loss)
                    elif self.paras.multi_gpu:
                        loss = loss.sum()
                        loss.backward()
                    else:
                        loss.backward()

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
                            self.verbose('Error : grad norm is NaN @ step ' + str(self.global_step))
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                        if self.global_step % self.log_step == 0:
                            # Log
                            self.log.add_scalar('lr', self.optimizer.get_lr()[0], self.global_step)
                            self.log.add_scalar('loss', (loss.item() * self.gradient_accumulation_steps), self.global_step)
                            self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                            progress.set_description("Loss %.4f" % (loss.item() * self.gradient_accumulation_steps))

                        if self.global_step % self.save_step == 0:
                            self.save_model('states')
                            mask_spec = self.up_sample_frames(spec_masked[0], return_first=True)
                            pred_spec = self.up_sample_frames(pred_spec[0], return_first=True)
                            true_spec = self.up_sample_frames(spec_stacked[0], return_first=True)
                            mask_spec = plot_spectrogram_to_numpy(mask_spec.data.cpu().numpy())
                            pred_spec = plot_spectrogram_to_numpy(pred_spec.data.cpu().numpy())
                            true_spec = plot_spectrogram_to_numpy(true_spec.data.cpu().numpy())
                            self.log.add_image('mask_spec', mask_spec, self.global_step)
                            self.log.add_image('pred_spec', pred_spec, self.global_step)
                            self.log.add_image('true_spec', true_spec, self.global_step)
                
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
        self.reset_train()
        

    def test_reconstruct(self):
        head_mask = None
        prune_headids = self.transformer_config['prune_headids']
        if prune_headids is not None:
            layer_num = self.transformer_config['num_hidden_layers']
            head_num = self.transformer_config['num_attention_heads']
            head_mask = torch.ones(layer_num, head_num)
            layer_ids = [idx // head_num for idx in prune_headids]
            head_ids = [idx % head_num for idx in prune_headids]
            head_mask[layer_ids, head_ids] = 0.0  

        epoch_loss = []
        for batch_is_valid, *batch in tqdm(self.dataloader):
            assert batch_is_valid

            spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = self.process_data(batch)
            if head_mask is not None:
                head_mask = head_mask.to(spec_masked.device)
            
            with torch.no_grad():
                loss, pred_spec = self.model(spec_masked, pos_enc, mask_label, attn_mask, spec_stacked, head_mask)
            epoch_loss.append(loss.item())

        print(sum(epoch_loss) / len(epoch_loss))


##########
# TESTER #
##########
class Tester(Solver):
    ''' Handler for complete testing progress'''
    def __init__(self, config, paras):
        super(Tester, self).__init__(config, paras)
        self.dump_dir = str(self.ckpt.split('.')[0]) + '-dump/'
        self.duo_feature = False # Set duo feature to False since only input mel is needed during testing
        self.load = True # Tester will load pre-trained models automatically


    def process_MAM_data(self, spec):
        """Process testing data for the masked acoustic model"""
        
        # Hack bucket if spec is loaded from the dataloader
        if len(spec.shape) == 4: # Bucketing should cause acoustic feature to have shape 1xBxTxD
            spec = spec.squeeze(0)
        # add arbitary batch axis B if input `spec` has shape of TxD
        elif len(spec.shape) == 2:
            spec = spec.unsqueeze(0)
        # input `spec` should have shape BxTxD
        elif len(spec.shape) != 3:
            raise ValueError('Input argument `spec` has invalid shape: {}'.format(spec.shape))

        # Down sample
        spec_stacked = self.down_sample_frames(spec) # (batch_size, seq_len, mel_dim * dr)

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = fast_position_encoding(seq_len, self.hidden_size) # (seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(spec_stacked)):
            attn_mask[idx][spec_len[idx]:] = 0 

        spec_stacked = spec_stacked.to(device=self.device, dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(device=self.device, dtype=torch.float32).expand(spec_stacked.size(0), *pos_enc.size())
        attn_mask = torch.FloatTensor(attn_mask).to(device=self.device, dtype=torch.float32)
        return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)


    def process_data(self, spec):
        assert(len(spec) == 3), 'dataloader should return (spec_stacked, pos_enc, attn_mask)'
        # Unpack and Hack bucket: Bucketing should cause acoustic feature to have shape 1xBxTxD'
        spec_stacked = spec[0].squeeze(0)
        pos_enc = spec[1].squeeze(0)
        attn_mask = spec[2].squeeze(0)
    
        spec_stacked = spec_stacked.to(device=self.device)
        if pos_enc.dim() == 3:
            # pos_enc: (batch_size, seq_len, hidden_size)
            # GPU memory need (batch_size * seq_len * hidden_size)
            pos_enc = torch.FloatTensor(pos_enc).to(device=self.device)
        elif pos_enc.dim() == 2:
            # pos_enc: (seq_len, hidden_size)
            # GPU memory only need (seq_len * hidden_size) even after expanded
            pos_enc = torch.FloatTensor(pos_enc).to(device=self.device).expand(spec_stacked.size(0), *pos_enc.size())
        attn_mask = torch.FloatTensor(attn_mask).to(device=self.device)
        return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)


    def tile_representations(self, reps):
        """ 
            Tile up the speech representations to match the amount of input frames.
            Input - encoded_layers shape: (num_hidden_layers, batch_size, sequence_length, hidden_size)
            Output - tiled_encoded_layers shape: (num_hidden_layers, batch_size, sequence_length * downsample_rate, hidden_size)
        """
        if len(reps.shape) == 3:
            reps = reps.unsqueeze(0)
        elif len(reps.shape) != 4:
            raise ValueError('Input argument `reps` has invalid shape: {}'.format(reps.shape))

        tiled_reps = reps.repeat(1, 1, 1, self.dr)
        tiled_reps = tiled_reps.reshape(reps.size(0), reps.size(1), reps.size(2)*self.dr, reps.size(3))

        # return the only layer if only one layer is given at input
        # else return representations of all layers
        if len(tiled_reps) == 1:
            return tiled_reps.squeeze(0) # (batch_size, sequence_length * downsample_rate, hidden_size)
        return tiled_reps # (num_hidden_layers, batch_size, sequence_length * downsample_rate, hidden_size)


    def plot(self, with_head=False):
        ''' Plotting the visualizations of the self-supervised pre-trained Transformer Model'''
        self.verbose('Testing set total ' + str(len(self.dataloader)) + ' batches.')
        if not os.path.exists(self.dump_dir): os.makedirs(self.dump_dir)
        with torch.no_grad():
            idx = 0
            for x in tqdm(self.dataloader, desc="Plotting"):
                spec_stacked, pos_enc, attn_mask = self.process_MAM_data(spec=x)
                
                if with_head:
                    pred_spec, _ = self.model(spec_stacked, pos_enc, attention_mask=attn_mask)
                    
                    # generate the model filled MAM spectrogram
                    spec_masked = copy.deepcopy(spec_stacked)
                    for i in range(len(spec_masked)):
                        sample_index = random.sample(range(len(spec_masked[i])), int(len(spec_masked[i])*self.transformer_config['mask_proportion']))
                        spec_masked[i][sample_index] = 0
                    fill_spec, _ = self.model(spec_masked, pos_enc, attention_mask=attn_mask)
                    
                    # plot reconstructed / ground-truth / MAM filled spectrogram
                    for y_pred, y_true, y_fill in zip(pred_spec, spec_stacked, fill_spec):
                        
                        y_pred = self.up_sample_frames(y_pred, return_first=True)
                        y_true = self.up_sample_frames(y_true, return_first=True)
                        y_fill = self.up_sample_frames(y_fill, return_first=True)
                        
                        plot_spectrogram(y_pred.data.cpu().numpy(), path=os.path.join(self.dump_dir, str(idx) + '_pred.png'))
                        plot_spectrogram(y_true.data.cpu().numpy(), path=os.path.join(self.dump_dir, str(idx) + '_true.png'))
                        plot_spectrogram(y_fill.data.cpu().numpy(), path=os.path.join(self.dump_dir, str(idx) + '_fill.png'))
                        
                        wave_pred = inv_spectrogram(y_pred.data.cpu().numpy().T)
                        wave_fill = inv_spectrogram(y_fill.data.cpu().numpy().T)
                        librosa.output.write_wav(os.path.join(self.dump_dir, str(idx) + '_pred.wav'), wave_pred, sample_rate)
                        librosa.output.write_wav(os.path.join(self.dump_dir, str(idx) + '_fill.wav'), wave_fill, sample_rate)
                        
                        idx += 1
                        if idx >= 10:
                            self.verbose('Spectrogram head generated samples are saved to: {}'.format(self.dump_dir))
                            exit() # visualize the first 10 testing samples
                else:
                    encoded_layers = self.transformer(spec_stacked, pos_enc, attention_mask=attn_mask, output_all_encoded_layers=True)
                    encoded_layers = torch.stack(encoded_layers)

                    layer_num = encoded_layers.size(0)
                    batch_size = encoded_layers.size(1)
                    seq_len = encoded_layers.size(2)
                    feature_dim = encoded_layers.size(3)

                    dckpt = torch.load(self.paras.load_ws)
                    weights = dckpt['Classifier']['weight']

                    flatten = encoded_layers.reshape(layer_num, -1)
                    weighted_sum = torch.matmul(weights[:layer_num], flatten).reshape(batch_size, seq_len, feature_dim)
                    # embeddings: (batch_size, seq_len, feature_dim)

                    targets = [encoded_layers[0], encoded_layers[-1], weighted_sum]
                    target_names = ['_hidden_first.png', '_hidden_last.png', '_hidden_weighted_sum.png']
                    for target, name in zip(targets, target_names):
                        for index, rep in enumerate(target):
                            if idx + index >= 10:
                                break
                            png_name = os.path.join(self.dump_dir, str(idx + index) + name)
                            self.verbose(f'Generating {png_name}')
                            plot_embedding(rep.data.cpu().numpy(), path=png_name)

                    idx += batch_size
                    if idx >= 10:
                        self.verbose('Model generated samples are saved to: {}'.format(self.dump_dir))
                        break # visualize the first 10 testing samples


    def plot_attention(self):
        attn_dir = os.path.join('attentions', self.exp_name)
        if not os.path.exists(attn_dir): os.makedirs(attn_dir)
        with torch.no_grad():
            sample_index = torch.arange(0, self.dataloader.dataset.__len__(), self.dataloader.dataset.__len__() // 5)
            for i, idx in enumerate(tqdm(sample_index)):
                x = self.dataloader.dataset[idx]
                spec_stacked, pos_enc, attn_mask = self.process_MAM_data(spec=x)
                
                all_attentions, _ = self.transformer(spec_stacked, pos_enc, attention_mask=attn_mask, output_all_encoded_layers=True)
                all_attentions = torch.stack(all_attentions).transpose(0, 1)
                # all_attentions: (batch_size, num_layer, num_head, Q_seq_len, K_seq_len)

                sample_dir = os.path.join(attn_dir, str(i))
                if not os.path.exists(sample_dir): os.makedirs(sample_dir)
                for layerid, layer_attentions in enumerate(all_attentions[0]):
                    for headid, head_attention in enumerate(layer_attentions):
                        plot_attention(head_attention.detach().cpu(), os.path.join(sample_dir, f'{layerid}-{headid}.png'))
                    plot_attention(layer_attentions.mean(dim=0).detach().cpu(), os.path.join(sample_dir, f'{layerid}-average.png'))


    def forward(self, spec, all_layers=True, tile=True, process_from_loader=False):
        """ 
            Generation of the Transformer Model Representation
            Input: A batch of spectrograms: (batch_size, seq_len, hidden_size)
            If `all_layers` == True:
                if `tile`: Output - A batch of representations: (batch_size, num_hiddem_layers, seq_len, hidden_size)
                if not `tile`: Output - A batch of representations: (batch_size, num_hiddem_layers, seq_len // downsample_rate, hidden_size)
            If `all_layers` == False:
                if `tile`: Output - A batch of representations: (batch_size, seq_len, hidden_size)
                if not `tile`: Output - A batch of representations: (batch_size, seq_len // downsample_rate, hidden_size)
            where `seq_len` is the sequence length of the input `spec`.
        """
            
        with torch.no_grad():
            
            if not process_from_loader:
                spec_stacked, pos_enc, attn_mask = self.process_MAM_data(spec=spec)
            else:
                spec_stacked, pos_enc, attn_mask = self.process_data(spec=spec) # Use dataloader to process MAM data to increase speed

            head_mask = None
            prune_headids = self.transformer_config['prune_headids']
            if prune_headids is not None:
                layer_num = self.transformer_config['num_hidden_layers']
                head_num = self.transformer_config['num_attention_heads']
                head_mask = torch.ones(layer_num, head_num).to(spec_stacked.device)
                layer_ids = [idx // head_num for idx in prune_headids]
                head_ids = [idx % head_num for idx in prune_headids]
                head_mask[layer_ids, head_ids] = 0.0  
            
            reps = self.transformer(spec_stacked, pos_enc, attention_mask=attn_mask, output_all_encoded_layers=all_layers, head_mask=head_mask)

            if type(reps) is list:
                reps = torch.stack(reps)
            # (num_hiddem_layers, batch_size, seq_len // downsample_rate, hidden_size) if `all_layers` or,
            # (batch_size, seq_len // downsample_rate, hidden_size) if not `all_layers`.

            # tile representations to match the input `seq_len` of `spec`
            if tile: reps = self.tile_representations(reps) # (num_hiddem_layers, batch_size, seq_len, hidden_size)
            
            if len(reps.shape) == 4: reps = reps.permute(1, 0, 2, 3).contiguous() # if `all_layers`: (batch_size, num_hidden_layers, -1, hidden_size)
            elif len(reps.shape) != 3: raise ValueError('Invalid representation shape!') # if not `all_layers`: (batch_size, -1, hidden_size)

        return reps


    def forward_with_head(self, spec, tile=True, process_from_loader=False):
        """ 
            Get representations from the spectrogram prediction head
            if `tile`: Output - A batch of representations: (batch_size, seq_len, hidden_size)
            if not `tile`: Output - A batch of representations: (batch_size, seq_len // downsample_rate, hidden_size)
        """
            
        with torch.no_grad():
            
            if not process_from_loader:
                spec_stacked, pos_enc, attn_mask = self.process_MAM_data(spec=spec)
            else:
                spec_stacked, pos_enc, attn_mask = self.process_data(spec=spec) # Use dataloader to process MAM data to increase speed
            _, reps = self.model(spec_stacked, pos_enc, attention_mask=attn_mask)

            # tile representations to match the input `seq_len` of `spec`
            if tile: reps = self.tile_representations(reps) # (batch_size, seq_len, hidden_size)

        return reps


    def forward_fine_tune(self, spec, tile=True, process_from_loader=False):
        """ 
            Fine tune the Transformer Model on downstream tasks
            Input: A batch of spectrograms: (batch_size, seq_len, hidden_size)
            Output - A batch of representations: (batch_size, seq_len, hidden_size)
            where `seq_len` is the sequence length of the input `spec`.
        """
            
        if not process_from_loader:
            spec_stacked, pos_enc, attn_mask = self.process_MAM_data(spec=spec)
        else:
            spec_stacked, pos_enc, attn_mask = self.process_data(spec=spec) # Use dataloader to process MAM data to increase speed
        reps = self.transformer(spec_stacked, pos_enc, attention_mask=attn_mask, output_all_encoded_layers=False)
        # reps: (batch_size, seq_len // downsample_rate, hidden_size)

        # tile representations to match the input `seq_len` of `spec`
        if tile: reps = self.tile_representations(reps) # (batch_size, seq_len, hidden_size)
        return reps