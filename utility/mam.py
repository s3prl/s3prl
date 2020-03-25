# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utils/mam.py ]
#   Synopsis     [ Moasked Acoustic Model data processing for the mockingjay model ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import copy
import random
import torch
import numpy as np
from functools import lru_cache


############
# CONSTANT #
############
DR = 3
HIDDEN_SIZE = 768
MASK_PROPORTION = 0.15
MASK_CONSECUTIVE = 1
MASK_FREQUENCY = 8
NOISE_PROPORTION = 0.15
MAX_SEQLEN = 3000


def down_sample_frames(spec, dr):
    left_over = spec.shape[1] % dr
    if left_over != 0: spec = spec[:, :-left_over, :]
    spec_stacked = spec.view(spec.shape[0], spec.shape[1]//dr, spec.shape[2]*dr)
    return spec_stacked


@lru_cache(maxsize=1)
def get_sinusoid_table(hidden_size):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)
        
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(hidden_size)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(MAX_SEQLEN)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def position_encoding(seq_len, hidden_size, batch_size=None, padding_idx=None):
    ''' position encoding table '''
    assert seq_len <= MAX_SEQLEN, f'constant MAX_SEQLEN ({MAX_SEQLEN}) in mam.py < received seq_len ({seq_len})'        
    table = get_sinusoid_table(hidden_size)[:seq_len]

    if padding_idx is not None:
        # deepcopy will slow down whole process when positional table is too large
        # this path is dreprecated and should never be used
        table = copy.deepcopy(table)
        table[padding_idx:] = 0. # zero vector for padding dimension

    if batch_size is not None:
        # using expand will not cause extra CPU memory allocation issue
        # however, the expanded tensor after put into GPU still need
        # GPU memory of expanded size, which should be avoided when
        # positional table is large
        # this path is not recommended
        batch_table = table.expand(batch_size, -1, -1)
        return batch_table # (batch_size, seq_len, hidden_size)
    else:
        # this path is most recommended, no extra CPU and GPU memory allocation
        # after getting the (seq_len, hidden_size) tensor, one should first put
        # this tensor into GPU then expand it
        return table  # (seq_len, hidden_size)


def process_train_MAM_data(spec, config=None):
    """Process training data for the masked acoustic model"""

    dr = config['downsample_rate'] if config is not None else DR
    hidden_size = config['hidden_size'] if config is not None else HIDDEN_SIZE
    mask_proportion = config['mask_proportion'] if config is not None else MASK_PROPORTION
    mask_consecutive = config['mask_consecutive'] if config is not None else MASK_CONSECUTIVE
    mask_frequency = config['mask_frequency'] if config is not None else MASK_FREQUENCY
    noise_proportion = config['noise_proportion'] if config is not None else NOISE_PROPORTION

    with torch.no_grad():
        if len(spec) == 2: # if self.duo_feature: dataloader will output `source_spec` and `target_spec`
            source_spec = spec[0]
            target_spec = spec[1]
        elif len(spec) == 1:
            source_spec = spec[0]
            target_spec = copy.deepcopy(spec[0])
        else:
            raise NotImplementedError('Input spec sould be either (spec,) or (target_spec, source_spec), where `spec` has shape BxTxD.')

        # Down sample
        spec_masked = down_sample_frames(source_spec, dr) # (batch_size, seq_len, mel_dim * dr)
        spec_stacked = down_sample_frames(target_spec, dr) # (batch_size, seq_len, mel_dim * dr)
        assert(spec_masked.shape[1] == spec_stacked.shape[1]), 'Input and output spectrogram should have the same shape'

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]
        
        pos_enc = position_encoding(seq_len, hidden_size) # (seq_len, hidden_size)
        mask_label = np.zeros_like(spec_stacked)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        for idx in range(batch_size):
            def starts_to_intervals(starts, consecutive):
                tiled = starts.expand(consecutive, starts.size(0)).T
                offset = torch.arange(consecutive).expand_as(tiled)
                intervals = tiled + offset
                return intervals.view(-1)
            
            valid_start_max = int(spec_len[idx] - mask_consecutive - 1) # compute max valid start point for a consecutive mask
            proportion = int(spec_len[idx] * mask_proportion // mask_consecutive)
            chosen_starts = torch.randperm(valid_start_max)[:proportion] # draw `proportion` samples from the range (0, valid_index_range) and without replacement
            chosen_intervals = starts_to_intervals(chosen_starts, mask_consecutive)
            
            # determine whether to mask / random / or do nothing to the frame
            dice = torch.rand(1).data.cpu()
            # mask to zero
            if bool(dice < 0.8):
                spec_masked[idx, chosen_intervals, :] = 0
            # replace to random frames
            elif bool(dice >= 0.8) and bool(dice < 0.9):
                random_starts = torch.randperm(valid_start_max)[:proportion]
                random_intervals = starts_to_intervals(random_starts, mask_consecutive)
                spec_masked[idx, chosen_intervals, :] = spec_masked[idx, random_intervals, :]
            # do nothing
            else:
                pass

            # frequency masking
            if mask_frequency > 0:
                rand_bandwidth = int(torch.randperm(mask_frequency).data.cpu().numpy()[:1][0])
                chosen_start = int(torch.randperm(spec_stacked.shape[2]-rand_bandwidth).data.cpu().numpy()[:1][0])
                spec_masked[idx, :, chosen_start:chosen_start+rand_bandwidth] = 0
                
            # noise augmentation
            dice = torch.rand(1).data.cpu()
            if bool(dice < noise_proportion):
                noise = np.random.normal(0, 0.1, spec_masked.shape)
                spec_masked += torch.FloatTensor(noise)

            # the gradients will be calculated on all chosen frames
            mask_label[idx, chosen_intervals, :] = 1
            if mask_frequency > 0:
                mask_label[idx, :, chosen_start:chosen_start+rand_bandwidth] = 1

            # zero vectors for padding dimension
            attn_mask[idx][spec_len[idx]:] = 0

        spec_masked = spec_masked.to(dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(dtype=torch.float32)
        mask_label = torch.BoolTensor(mask_label).to(dtype=torch.bool)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)
        spec_stacked = spec_stacked.to(dtype=torch.float32)

    return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked


def process_test_MAM_data(spec, config=None):
    """Process testing data for the masked acoustic model"""
    
    dr = config['downsample_rate'] if config is not None else DR
    hidden_size = config['hidden_size'] if config is not None else HIDDEN_SIZE

    with torch.no_grad():
        if len(spec) != 1:
            raise NotImplementedError('Input spec sould be a tuple of: (spec,), where `spec` has shape BxTxD.')

        # Down sample
        spec_stacked = down_sample_frames(spec[0], dr) # (batch_size, seq_len, mel_dim * dr)

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = position_encoding(seq_len, hidden_size) # (seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(spec_stacked)):
            attn_mask[idx][spec_len[idx]:] = 0 

        spec_stacked = spec_stacked.to(dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(dtype=torch.float32)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)

    return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)