# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pytorch_kaldi/nn_transformer.py ]
#   Synopsis     [ wrapper class for downstream feature extraction or finetune ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from functools import lru_cache
from distutils.util import strtobool
from transformer.model import TransformerConfig, TransformerModel


###############
# TRANSFORMER #
###############
"""
Use this class to extract features from the Transformer model,
or to finetune the pre-trained Transformer with any downstream tasks.
Also, this class is `pytorch-kaldi` ready,
hence we need to use `str` instead of `bool` in the options dict,
as pytorch-kaldi scripts will pass in str.

Params:
    `options`: a python dictionary containing the following keys:
        ckpt_file: str, a path specifying the pre-trained ckpt file
        load_pretrain: str, ['True', 'False'], whether to load pre-trained weights
        no_grad: str, ['True', 'False'], whether to have gradient flow over this class
        dropout: float/str, use float to modify dropout value during downstream finetune, or use the str `default` for pre-train default values
        spec_aug: str, ['True', 'False'], whether to apply SpecAugment on inputs (used for ASR training)
        spec_aug_prev: str, ['True', 'False'], apply spec augment on input acoustic features if True, else apply on output representations (used for ASR training)
        weighted_sum: str, ['True', 'False'], whether to use a learnable weighted sum to integrate hidden representations from all layers, if False then use the last
        select_layer: int, select from all hidden representations, set to -1 to select the last (will only be used when weighted_sum is False)
    `intput_dim`: int, input dimension of model

An example `options` dictionary:
options = {
    'ckpt_file'     : './result/result_transformer/libri_sd1337_fmllrBase960-F-N-K-RA/states-1000000.ckpt',
    'load_pretrain' : 'True',
    'no_grad'       : 'True',
    'dropout'       : 'default',
    'spec_aug'      : 'False',
    'spec_aug_prev' : 'True',
    'weighted_sum'  : 'False',
    'select_layer'  : -1,
}
"""
class TRANSFORMER(nn.Module):
    def __init__(self, options, inp_dim, config=None):
        super(TRANSFORMER, self).__init__()

        if config is not None:
            self.config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
        else:
            all_states = torch.load(options["ckpt_file"], map_location='cpu')
            self.config = all_states['Settings']['Config']

        self.no_grad = bool(strtobool(options['no_grad']))
        self.spec_aug = bool(strtobool(options['spec_aug']))
        self.spec_aug_prev = bool(strtobool(options['spec_aug_prev']))
        self.weighted_sum = bool(strtobool(options['weighted_sum']))
        self.select_layer = int(options['select_layer'])
        if (not self.no_grad) and (not self.spec_aug_prev): raise RuntimeError('Only one of them can be set False!')
        
        # increase dropout
        if str(options['dropout']) != 'default':
            self.config['transformer']['hidden_dropout_prob'] = float(options['dropout'])
            self.config['transformer']['attention_probs_dropout_prob'] = float(options['dropout'])

        # Model Config
        self.model_config = TransformerConfig(self.config)
        self.dr = self.model_config.downsample_rate
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        if not (self.select_layer in list(range(-1, self.num_layers))): raise RuntimeError('Out of range int for \'select_layer\'!')

        # use weighted sum from all layers
        if self.weighted_sum:
            self.weight = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)

        # Build model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = TransformerModel(self.model_config, inp_dim).to(self.device)
        self.model.eval() if self.no_grad else self.model.train()
        
        # Load from a PyTorch state_dict
        load = bool(strtobool(options["load_pretrain"]))
        if load: 
            self.load_model(all_states['Transformer'])
            print('[Transformer] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        
        self.out_dim = self.hidden_size # 768, This attribute is for pytorch-kaldi and downstream runner
        self.permute_input = True # This attribute is for the forward method. If Ture then input ouput is in the shape of (T, B, D), if False then in (B, T, D)


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
                print('Weights of {} not initialized from pretrained model: {}'.format(
                    self.model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                print('Weights from pretrained model not used in {}: {}'.format(
                    self.model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                    self.model.__class__.__name__, '\n\t'.join(error_msgs)))
            print('[Transformer] - Pre-trained weights loaded!')

        except: print('[Transformer] - Pre-trained weights NOT loaded!')


    def down_sample_frames(self, spec):
        spec = spec.contiguous()
        left_over = spec.shape[1] % self.dr
        if left_over != 0: spec = spec[:, :-left_over, :]
        spec_stacked = spec.view(spec.shape[0], spec.shape[1]//self.dr, spec.shape[2]*self.dr)
        return spec_stacked
        

    def process_input_data(self, spec):
        """Process input data for the model"""
        
        # add arbitary batch axis B if input `spec` has shape of TxD
        if len(spec.shape) == 2:
            spec = spec.unsqueeze(0)
        # input `spec` should have shape BxTxD
        elif len(spec.shape) != 3:
            raise ValueError('Input argument `spec` has invalid shape: {}'.format(spec.shape))

        # Down sample
        if self.dr > 1:
            spec_stacked = self.down_sample_frames(spec) # (batch_size, seq_len, feature_dim * dr)
        else:
            spec_stacked = spec

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.cpu().data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = position_encoding(seq_len, self.hidden_size) # (seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(spec_stacked)):
            attn_mask[idx][spec_len[idx]:] = 0 

        if self.spec_aug and self.spec_aug_prev and self.model.training:
            spec_stacked = spec_augment(spec_stacked, mask_T=70, mask_F=4, num_T=2, num_F=2, p=1.0) # (batch_size, seq_len, feature_dim * dr)
        spec_stacked = spec_stacked.to(device=self.device, dtype=torch.float32) # (batch_size, seq_len, feature_dim * dr)
        pos_enc = torch.FloatTensor(pos_enc).to(device=self.device, dtype=torch.float32).expand(spec_stacked.size(0), *pos_enc.size()) # (batch_size, seq_len, hidden_size)
        attn_mask = torch.FloatTensor(attn_mask).to(device=self.device, dtype=torch.float32) # (batch_size, seq_len)
        return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)


    def tile_representations(self, reps):
        """ 
        Tile up the speech representations to match the amount of input frames.
        Input - encoded_layers shape: (batch_size, sequence_length, hidden_size)
        Output - tiled_encoded_layers shape: (batch_size, sequence_length * downsample_rate, hidden_size)
        """
        if len(reps.shape) != 3:
            raise ValueError('Input argument `reps` has invalid shape: {}'.format(reps.shape))

        tiled_reps = reps.repeat(1, 1, self.dr)
        tiled_reps = tiled_reps.reshape(reps.size(0), reps.size(1)*self.dr, reps.size(2))
        return tiled_reps # (batch_size, sequence_length * downsample_rate, hidden_size)
        

    def _forward(self, x):

        if self.permute_input:
            x = x.permute(1, 0, 2).contiguous() # (T, B, D) -> (B, T, D)
            input_len = x.shape[0]
        else:
            input_len = x.shape[1]

        # Compute padding to compromise the downsample loss
        left_over = input_len % self.dr
        if left_over % 2 == 0:
            left_pad = left_over // 2
            right_pad = left_pad
        else:
            left_pad = left_over // 2
            right_pad = left_over // 2 + 1

        # Model forwarding
        spec_stacked, pos_enc, attn_mask = self.process_input_data(x) # x shape: (B, T, D)
        x = self.model(spec_stacked, pos_enc, attn_mask, output_all_encoded_layers=self.weighted_sum or self.select_layer != -1) # (B, T, D) or # (N, B, T, D)

        # Apply weighted sum
        if self.weighted_sum:
            if type(x) is list: x = torch.stack(x)
            softmax_weight = nn.functional.softmax(self.weight, dim=-1)
            B, T, D = x.shape[1], x.shape[2], x.shape[3]
            x = x.reshape(self.num_layers, -1)
            x = torch.matmul(softmax_weight, x).reshape(B, T, D)
        # Select a specific layer
        elif self.select_layer != -1:
            x = x[self.select_layer]

        if self.spec_aug and not self.spec_aug_prev and self.model.training:
            x = spec_augment(x, mask_T=70, mask_F=86, num_T=2, num_F=2, p=1.0) # (B, T, D)

        # If using a downsampling model, apply tile and padding
        if x.shape[1] != input_len:
            x = self.tile_representations(x)

            # padding
            x = x.permute(0, 2, 1).contiguous() # (B, T, D) -> (B, D, T)
            padding = nn.ReplicationPad1d((left_pad, right_pad))
            x = padding(x)
            
            if self.permute_input: x = x.permute(2, 0, 1).contiguous() # (B, D, T) -> (T, B, D)
            else: x = x.permute(0, 2, 1).contiguous() # (B, D, T) -> (B, T, D)
        
        # If not using a downsampling model, permute to output
        elif self.permute_input:
            x = x.permute(1, 0, 2).contiguous() # (B, T, D) -> (T, B, D)
        
        # else: (B, T, D)
        return x


    def forward(self, x):
        if self.no_grad:
            with torch.no_grad():
                self.model.eval()
                x = self._forward(x)
        else:
            x = self._forward(x)
        return x


#######################
# POSITIONAL ENCODING #
#######################
MAX_SEQLEN = 5000
@lru_cache(maxsize=128)
def get_sinusoid_table(hidden_size):
    def _cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)
    def _get_posi_angle_vec(position):
        return [_cal_angle(position, hid_j) for hid_j in range(hidden_size)]
    sinusoid_table = np.array([_get_posi_angle_vec(pos_i) for pos_i in range(MAX_SEQLEN)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def position_encoding(seq_len, hidden_size):
    """ position encoding table """     
    table = get_sinusoid_table(hidden_size)[:seq_len]
    # no extra CPU and GPU memory allocation
    # after getting the (seq_len, hidden_size) tensor, one should first put
    # this tensor into GPU then expand it
    return table  # (seq_len, hidden_size)


################
# SPEC AUGMENT #
################
"""
Process training data for the supervised ASR model by 
masking to time-steps and channels during training
which delays overfitting and significantly improves the final accuracy numbers.
Input:
    `spec`: input real frames, with shape: (batch_size, seq_len, feature_dim)
    `mask_T`: the time mask parameter T described in the SpecAugment paper, 
              we use default values based on the LD Policy
              (In paper: T=100, we use 70 since we are training on the 100 hr subset only)
    `mask_F`: the frequency mask parameter F described in the SpecAugment paper, 
              we use default values based on the LD Policy
              (In paper: F=27:D=80*3 -> F=4.5:D=40, where D is acoustic dimension)
    `num_T` : the number of time masks applied (In paper: mT=2)
    `num_F` : the number of frequency masks applied (In paper: mF=2)
    `p` : upper bound ratio (In paper: p=1.0)
Output:
    `spec`: augmented frames, with shape: (batch_size, seq_len, feature_dim)
"""
def spec_augment(spec, mask_T=70, mask_F=4, num_T=2, num_F=2, p=1.0):

    def _start_to_intervals(starts, consecutive):
        tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
        offset = torch.arange(consecutive).expand_as(tiled)
        intervals = tiled + offset
        return intervals.view(-1)

    with torch.no_grad():
        upper_bound = spec.shape[1] * p # upper bound on the time mask so that a time mask cannot be wider than p times the number of time steps
        
        for idx in range(spec.shape[0]):

            # time masking
            if mask_T > 0 and mask_T < upper_bound:
                for _ in range(num_T):
                    rand_consecutive = random.randint(0, mask_T)
                    chosen_start = torch.randperm(spec.shape[1] - rand_consecutive)[:1]
                    chosen_intervals = _start_to_intervals(chosen_start, rand_consecutive)
                    spec[idx, chosen_intervals, :] = 0

            # frequency masking
            if mask_F > 0:
                for _ in range(num_F):
                    rand_bandwidth = random.randint(0, mask_F)
                    chosen_start = torch.randperm(spec.shape[2] - rand_bandwidth)[:1]
                    chosen_intervals = _start_to_intervals(chosen_start, rand_bandwidth)
                    spec[idx, :, chosen_intervals] = 0

        return spec


#######
# LIN #
#######
"""
Linear Input Networks (LIN) for domain adaptation
Params:
    `options`: a python dictionary containing arguments for pytorch kaldi, give None if not using with pytorch-kaldi:
    `intput_dim`: int, input dimension of model
"""
class LIN(nn.Module):
    def __init__(self, options, inp_dim):
        super(LIN, self).__init__()

        self.out_dim = inp_dim # This attribute is for pytorch-kaldi
        self.linear = nn.Linear(inp_dim, inp_dim)
        self.linear.weight.data.copy_(torch.eye(inp_dim))
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.linear = self.linear.to(self.device)
        self.linear.train()
        
    def forward(self, x):
        x = self.linear(x)
        return x