# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ transformer/nn_transformer.py ]
#   Synopsis     [ wrapper class for downstream feature extraction or finetune ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import copy
import math
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from functools import lru_cache
from distutils.util import strtobool
from utility.preprocessor import OnlinePreprocessor
from transformer.model import TransformerConfig, TransformerModel
from transformer.model import TransformerSpecPredictionHead
from transformer.model_dual import DualTransformerConfig
from transformer.model_dual import TransformerPhoneticEncoder, TransformerSpeakerEncoder


###############
# TRANSFORMER #
###############
class TransformerBaseWrapper(nn.Module):
    """ 
    A base class for all Transformer wrappers.
    Child classes only need to implement the __init__() and forward() method.
    """
    def __init__(self, options, inp_dim, config=None, online_config=None):
        super(TransformerBaseWrapper, self).__init__()

        # read config
        if config is not None:
            self.config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
        else:
            self.all_states = torch.load(options["ckpt_file"], map_location='cpu')
            self.config = self.all_states['Settings']['Config']

        # parse the options dict
        self.load = bool(strtobool(options["load_pretrain"]))
        self.no_grad = bool(strtobool(options['no_grad']))
        self.spec_aug = bool(strtobool(options['spec_aug']))
        self.spec_aug_prev = bool(strtobool(options['spec_aug_prev']))
        self.weighted_sum = bool(strtobool(options['weighted_sum']))
        self.select_layer = int(options['select_layer'])
        self.permute_input = bool(strtobool(options['permute_input']))
        if (not self.no_grad) and (not self.spec_aug_prev): raise RuntimeError('Only one of them can be set False!')
        if str(options['dropout']) != 'default': # increase dropout if specified
            self.config['transformer']['hidden_dropout_prob'] = float(options['dropout'])
            self.config['transformer']['attention_probs_dropout_prob'] = float(options['dropout'])

        # Set model config
        self.model_config = TransformerConfig(self.config)
        self.dr = self.model_config.downsample_rate
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.max_input_length = self.config['transformer']['max_input_length'] if 'max_input_length' in self.config['transformer'] else 0
        if online_config is not None: self.config['online'] = online_config
        if 'online' in self.config:
            preprocessor, inp_dim = self.get_preprocessor(self.config['online'])
            self.preprocessor = preprocessor
        self.inp_dim = inp_dim if inp_dim > 0 else self.config['transformer']['input_dim']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        if self.max_input_length > 0: print('[Transformer] - Maximum input length: ', self.max_input_length)
        if not (self.select_layer in list(range(-1, self.num_layers))): raise RuntimeError('Out of range int for \'select_layer\'!')
        if self.weighted_sum:
            self.weight = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)


    def load_model(self, transformer_model, state_dict):
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

            load(transformer_model)
            if len(missing_keys) > 0:
                print('Weights of {} not initialized from pretrained model: {}'.format(
                    transformer_model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                print('Weights from pretrained model not used in {}: {}'.format(
                    transformer_model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                    transformer_model.__class__.__name__, '\n\t'.join(error_msgs)))
            print('[Transformer] - Pre-trained weights loaded!')
            return transformer_model

        except: 
            raise RuntimeError('[Transformer] - Pre-trained weights NOT loaded!')

    def get_preprocessor(self, online_config):
        # load the same preprocessor as pretraining stage
        upstream_input_feat = online_config['input']
        upstream_input_feat['channel'] = 0
        upstream_target_feat = online_config['target']
        upstream_target_feat['channel'] = 0
        preprocessor = OnlinePreprocessor(**online_config, feat_list=[upstream_input_feat, upstream_target_feat])
        upstream_feat = preprocessor()[0]
        upstream_input_dim = upstream_feat.size(-1)
        return preprocessor, upstream_input_dim

    def down_sample_frames(self, spec):
        spec = spec.contiguous()
        left_over = spec.shape[1] % self.dr
        if left_over != 0: spec = spec[:, :-left_over, :]
        spec_stacked = spec.view(spec.shape[0], spec.shape[1]//self.dr, spec.shape[2]*self.dr)
        return spec_stacked
        

    def process_input_data(self, feat):
        """Process input data for the model"""
        
        # add arbitary batch axis B if input `feat` has shape of TxD
        if len(feat.shape) == 2:
            feat = feat.unsqueeze(0)
        # input `feat` should have shape BxTxD
        elif len(feat.shape) != 3:
            raise ValueError('Input argument `feat` has invalid shape: {}'.format(feat.shape))
        
        scale = 1 if not 'online' in self.config else \
                self.self.config['online']['sample_rate'] // 100 if self.config['online']['input']['feat_type'] == 'wav' else 1

        # Down sample
        if self.dr > 1 and self.inp_dim > 1: # no downsampling for waveform
            feat = self.down_sample_frames(feat) # (batch_size, seq_len, feature_dim * dr)

        # Record length for each uttr
        spec_len = (feat.sum(dim=-1) != 0).long().sum(dim=-1) // scale
        spec_len = spec_len.tolist()

        batch_size = feat.shape[0]
        seq_len = feat.shape[1] // scale

        pos_enc = position_encoding(seq_len, self.hidden_size) # (seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(feat)):
            attn_mask[idx][spec_len[idx]:] = 0 

        if self.spec_aug and self.spec_aug_prev and self.model.training and self.inp_dim > 1:
            feat = spec_augment(feat, mask_T=70, mask_F=4, num_T=2, num_F=2, p=1.0) # (batch_size, seq_len, feature_dim * dr)
        feat = feat.to(device=self.device, dtype=torch.float32) # (batch_size, seq_len, feature_dim * dr)
        pos_enc = torch.FloatTensor(pos_enc).to(device=self.device, dtype=torch.float32).expand(feat.size(0), *pos_enc.size()) # (batch_size, seq_len, hidden_size)
        attn_mask = torch.FloatTensor(attn_mask).to(device=self.device, dtype=torch.float32) # (batch_size, seq_len)
        return feat, pos_enc, attn_mask # (x, pos_enc, attention_mask)


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


    def upsample(self, x, input_len):
        # Compute padding to compromise the downsample loss
        left_over = input_len % self.dr
        if left_over % 2 == 0:
            left_pad = left_over // 2
            right_pad = left_pad
        else:
            left_pad = left_over // 2
            right_pad = left_over // 2 + 1
        
        x = self.tile_representations(x)

        # padding
        x = x.permute(0, 2, 1).contiguous() # (B, T, D) -> (B, D, T)
        padding = nn.ReplicationPad1d((left_pad, right_pad))
        x = padding(x)
        
        x = x.permute(0, 2, 1).contiguous() # (B, D, T) -> (B, T, D)
        return x


    def _forward(self, x):

        if self.permute_input:
            x = x.permute(1, 0, 2).contiguous() # (T, B, D) -> (B, T, D)
        input_len = x.shape[1]

        # forward the whole sequence at once
        if self.max_input_length == 0 or input_len <= self.max_input_length:
            feat, pos_enc, attn_mask = self.process_input_data(x) # x shape: (B, T, D)
            x = self.model(feat, pos_enc, attn_mask, output_all_encoded_layers=self.weighted_sum or self.select_layer != -1) # (B, T, D) or # (N, B, T, D)
        # forward the sequence in chunks then concat
        else:
            chunks = torch.chunk(x, chunks=math.ceil(input_len / self.max_input_length), dim=1)
            x_ = []
            for chunk in chunks:
                feat, pos_enc, attn_mask = self.process_input_data(chunk) # x shape: (B, T, D)
                chunk = self.model(feat, pos_enc, attn_mask, output_all_encoded_layers=self.weighted_sum or self.select_layer != -1) # (B, T, D) or # (N, B, T, D)
                x_.append(torch.stack(chunk) if type(chunk) is list else chunk)
            x = torch.cat(x_, dim=2 if (self.weighted_sum or self.select_layer != -1) else 1)

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

        if self.spec_aug and not self.spec_aug_prev and self.model.training and self.inp_dim > 1:
            x = spec_augment(x, mask_T=70, mask_F=86, num_T=2, num_F=2, p=1.0) # (B, T, D)

        # If using a downsampling model, apply tile and padding
        if self.dr > 1:
            x = self.upsample(x, input_len) # (B, T, D)
        
        # permute to output
        if self.permute_input:
            x = x.permute(1, 0, 2).contiguous() # (B, T, D) -> (T, B, D)

        return x # (B, T, D) or (T, B, D)


###############
# TRANSFORMER #
###############
class TRANSFORMER(TransformerBaseWrapper):
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
            permute_input: str, ['True', 'False'], this attribute is for the forward method. If Ture then input ouput is in the shape of (T, B, D), if False then in (B, T, D)
        `intput_dim`: int, input dimension of model
        `config`: optional, reads the given yaml config and not use the config stored in `ckpt_file`

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
        'permute_input' : 'False',
    }
    """
    def __init__(self, options, inp_dim, config=None, online_config=None):
        super(TRANSFORMER, self).__init__(options, inp_dim, config, online_config)

        # Build model
        self.model = TransformerModel(self.model_config, self.inp_dim).to(self.device)
        self.model.eval() if self.no_grad else self.model.train()
        self.out_dim = self.hidden_size # This attribute is necessary, for pytorch-kaldi and run_downstream.py
        
        # Load from a PyTorch state_dict
        if self.load: 
            self.model = self.load_model(self.model, self.all_states['Transformer'])
            print('[Transformer] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))


    def forward(self, x):
        if hasattr(self, 'preprocessor'):
            x = self.preprocessor(x.transpose(1, 2).contiguous())[0]
        if self.no_grad:
            with torch.no_grad():
                x = self._forward(x)
        else:
            x = self._forward(x)
        return x


####################
# SPEC TRANSFORMER #
####################
class SPEC_TRANSFORMER(TRANSFORMER):
    def __init__(self, options, inp_dim, config=None, online_config=None):
        super(SPEC_TRANSFORMER, self).__init__(options, inp_dim, config, online_config)

        # build head
        self.SpecHead = TransformerSpecPredictionHead(self.model_config, inp_dim).to(self.device)
        self.SpecHead.eval() if self.no_grad else self.SpecHead.train()
        
        # Load from a PyTorch state_dict
        if self.load:
            self.SpecHead.load_state_dict(self.all_states['SpecHead'])
            print('[Spec Transformer] - Number of parameters: ' + str(sum(p.numel() for p in self.SpecHead.parameters() if p.requires_grad)))

    def forward(self, x):
        if hasattr(self, 'preprocessor'):
            x = self.preprocessor(x.transpose(1, 2).contiguous())[0]
        if self.no_grad:
            with torch.no_grad():
                x = self._forward(x)
                x, _ = self.SpecHead(x)
        else:
            x = self._forward(x)
            x, _ = self.SpecHead(x)
        return x


####################
# DUAL TRANSFORMER #
####################
class DUAL_TRANSFORMER(TransformerBaseWrapper):
    def __init__(self, options, inp_dim, config=None, mode='phone', with_recognizer=True):
        super(DUAL_TRANSFORMER, self).__init__(options, inp_dim, config)

        del self.model_config
        self.model_config = DualTransformerConfig(self.config)
        self.out_dim = 0 # This attribute is necessary, for pytorch-kaldi and run_downstream.py
        self.mode = mode # can be 'phone', 'speaker', or 'phone speaker'
        assert self.mode in 'phone speaker'
        
        # Build model
        if 'phone' in self.mode:
            self.PhoneticTransformer = TransformerPhoneticEncoder(self.model_config, self.inp_dim, with_recognizer=with_recognizer).to(self.device)
            self.PhoneticTransformer.eval() if self.no_grad else self.PhoneticTransformer.train()
            self.model = self.PhoneticTransformer
        
        if 'speaker' in self.mode:
            self.SpeakerTransformer = TransformerSpeakerEncoder(self.model_config, self.inp_dim, with_recognizer=with_recognizer).to(self.device)
            self.SpeakerTransformer.eval() if self.no_grad else self.SpeakerTransformer.train()
            self.model = self.SpeakerTransformer
        
        # Load from a PyTorch state_dict
        load = bool(strtobool(options["load_pretrain"]))
        if load and 'phone' in self.mode:
            self.PhoneticTransformer.Transformer = self.load_model(self.PhoneticTransformer.Transformer, 
                                                                   self.all_states['PhoneticTransformer'])
            if hasattr(self.PhoneticTransformer, 'PhoneRecognizer'): 
                self.PhoneticTransformer.PhoneRecognizer.load_state_dict(self.all_states['PhoneticLayer'])
            self.out_dim += self.PhoneticTransformer.out_dim
            print('[Phonetic Transformer] - Number of parameters: ' + str(sum(p.numel() for p in self.PhoneticTransformer.parameters() if p.requires_grad)))

        if load and 'speaker' in self.mode:
            self.SpeakerTransformer.Transformer = self.load_model(self.SpeakerTransformer.Transformer, 
                                                                   self.all_states['SpeakerTransformer'])
            if hasattr(self.SpeakerTransformer, 'SpeakerRecognizer'):
                self.SpeakerTransformer.SpeakerRecognizer.load_state_dict(self.all_states['SpeakerLayer'])
            self.out_dim += self.SpeakerTransformer.out_dim
            print('[Speaker Transformer] - Number of parameters: ' + str(sum(p.numel() for p in self.SpeakerTransformer.parameters() if p.requires_grad)))


    def _dual_forward(self, x):
        if hasattr(self, 'preprocessor'):
            x = self.preprocessor(x.transpose(1, 2).contiguous())[0]
        if 'phone' in self.mode and 'speaker' in self.mode:
            self.model = self.PhoneticTransformer
            phonetic_code = self._forward(copy.deepcopy(x))
            self.model = self.SpeakerTransformer
            speaker_code = self._forward(x)
            if self.model_config.average_pooling: 
                speaker_code = speaker_code.repeat(1, phonetic_code.size(1), 1)
            if self.model_config.combine == 'concat':
                x = torch.cat((phonetic_code, speaker_code), dim=2)
            elif self.model_config.combine == 'add':
                x = phonetic_code + speaker_code
            else:
                raise NotImplementedError

        elif ('phone' in self.mode) != ('speaker' in self.mode): # exclusive or
            x = self._forward(x)
        else:
            raise NotImplementedError
        return x


    def forward(self, x):
        if self.no_grad:
            with torch.no_grad():
                x = self._dual_forward(x)
        else:
            x = self._dual_forward(x)
        return x


#######################
# POSITIONAL ENCODING #
#######################
MAX_SEQLEN = 5000
@lru_cache(maxsize=1)
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