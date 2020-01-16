# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ mockingjay/nn_mockingjay.py ]
#   Synopsis     [ wrapper class for downstream feature extraction or finetune ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import sys
sys.path.append('/media/andi611/1TBSSD/Mockingjay-Speech-Representation')
import torch
import numpy as np
import torch.nn as nn
from distutils.util import strtobool
from mockingjay.model import MockingjayConfig, MockingjayModel


##############
# MOCKINGJAY #
##############
"""
Use this class to extract features from the Mockingjay model,
or to finetune the pre-trained Mockingjay with any downstream tasks.
Also, this class is `pytorch-kaldi` ready.

Params:
    `options`: a python dictionary containing the following keys:
        ckpt_file: str, a path specifying the pre-trained ckpt file
        load_pretrain: bool, whether to load pre-trained weights
        no_grad: bool, whether to have gradient flow over this class
        dropout: float/str, use float to modify dropout value during downstream finetune, or use the str `default` for pre-train default values
    `intput_dim`: int, input dimension of model

An example `options` dictionary:
    ckpt_file = /media/andi611/1TBSSD/Mockingjay-Speech-Representation/result/result_mockingjay/mockingjay_libri_sd1337_MelBase460-K/mockingjay-500000.ckpt
    load_pretrain = True
    no_grad = False
    dropout = default
"""
class MOCKINGJAY(nn.Module):
    def __init__(self, options, inp_dim):
        super(MOCKINGJAY, self).__init__()
        
        all_states = torch.load(options["ckpt_file"], map_location='cpu')
        self.config = all_states['Settings']['Config']
        self.no_grad = bool(strtobool(options["no_grad"]))

        # increase dropout
        if str(options['dropout']) != 'default':
            self.config['mockingjay']['hidden_dropout_prob'] = float(options['dropout'])
            self.config['mockingjay']['attention_probs_dropout_prob'] = float(options['dropout'])

        # Model Config
        self.model_config = MockingjayConfig(self.config)
        self.dr = self.model_config.downsample_rate
        self.hidden_size = self.model_config.hidden_size

        # Build model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = MockingjayModel(self.model_config, inp_dim).to(self.device)
        
        # Load from a PyTorch state_dict
        load = bool(strtobool(options["load_pretrain"]))
        if load: 
            self.load_model(all_states['Mockingjay'])
            print('[Mockingjay] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        
        self.out_dim = 768 # This attribute is for pytorch-kaldi


    def load_model(self, state_dict):
        try:
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

            load(self.model)
            if len(missing_keys) > 0:
                print("Weights of {} not initialized from pretrained model: {}".format(
                    self.model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                print("Weights from pretrained model not used in {}: {}".format(
                    self.model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                    self.model.__class__.__name__, "\n\t".join(error_msgs)))
            print('[Mockingjay] - Pre-trained weights loaded!')

        except: print('[Mockingjay] - Pre-trained weights NOT loaded!')


    def down_sample_frames(self, spec):
        spec = spec.contiguous()
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


    def process_MAM_data(self, spec):
        """Process testing data for the masked acoustic model"""
        
        # add arbitary batch axis B if input `spec` has shape of TxD
        if len(spec.shape) == 2:
            spec = spec.unsqueeze(0)
        # input `spec` should have shape BxTxD
        elif len(spec.shape) != 3:
            raise ValueError('Input argument `spec` has invalid shape: {}'.format(spec.shape))

        # Down sample
        spec_stacked = self.down_sample_frames(spec) # (batch_size, seq_len, mel_dim * dr)

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.cpu().data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = self.position_encoding(seq_len, batch_size) # (batch_size, seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(spec_stacked)):
            pos_enc[idx][spec_len[idx]:] = 0  
            attn_mask[idx][spec_len[idx]:] = 0 

        spec_stacked = spec_stacked.to(device=self.device, dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(device=self.device, dtype=torch.float32)
        attn_mask = torch.FloatTensor(attn_mask).to(device=self.device, dtype=torch.float32)
        return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)


    def tile_representations(self, reps):
        """ 
            Tile up the mockingjay representations to match the amount of input frames.
            Input - encoded_layers shape: (batch_size, sequence_length, hidden_size)
            Output - tiled_encoded_layers shape: (batch_size, sequence_length * downsample_rate, hidden_size)
        """
        if len(reps.shape) != 3:
            raise ValueError('Input argument `reps` has invalid shape: {}'.format(reps.shape))

        tiled_reps = reps.repeat(1, 1, self.dr)
        tiled_reps = tiled_reps.reshape(reps.size(0), reps.size(1)*self.dr, reps.size(2))
        return tiled_reps # (batch_size, sequence_length * downsample_rate, hidden_size)
        

    def _forward(self, x):

        # Compute padding to compromise the downsample loss
        input_len = x.shape[0]
        left_over = input_len % self.dr

        if left_over % 2 == 0:
            left_pad = left_over // 2
            right_pad = left_pad
        else:
            left_pad = left_over // 2
            right_pad = left_over // 2 + 1

        # Model forwarding
        x = x.permute(1, 0, 2).contiguous() # (T, B, D) -> (B, T, D)
        spec_stacked, pos_enc, attn_mask = self.process_MAM_data(x)
        x = self.model(spec_stacked, pos_enc, attn_mask, output_all_encoded_layers=False) # (B, T, D)
        
        # If using a downsampling model, apply tile and padding
        if x.shape[1] != input_len:
            x = self.tile_representations(x)

            # padding
            x = x.permute(0, 2, 1).contiguous() # (B, T, D) -> (B, D, T)
            padding = nn.ReplicationPad1d((left_pad, right_pad))
            x = padding(x)
            x = x.permute(2, 0, 1).contiguous() # (B, D, T) -> (T, B, D)
        
        # If not using a downsampling model, permute to output
        else:
            x = x.permute(1, 0, 2).contiguous() # (B, T, D) -> (T, B, D)
        return x


    def forward(self, x):
        if self.no_grad:
            with torch.no_grad():
                self.model.eval()
                x = self._forward(x)
        else:
            x = self._forward(x)
        return x

