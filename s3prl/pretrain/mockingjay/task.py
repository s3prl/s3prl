# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/mockingjay/task.py ]
#   Synopsis     [ Masked Acoustic Model data processing for pre-training the transformer model ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import copy
import torch
import random
import numpy as np
from functools import lru_cache


############
# CONSTANT #
############
MAX_SEQLEN = 3000


@lru_cache(maxsize=128)
def get_sinusoid_table(hidden_size):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)
        
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(hidden_size)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(MAX_SEQLEN)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def fast_position_encoding(seq_len, hidden_size, batch_size=None, padding_idx=None):
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


def generate_masked_acoustic_model_data(spec, config):
    """Process training data for the masked acoustic model"""

    with torch.no_grad():

        # Start
        if len(spec) == 2: # if self.duo_feature: dataloader will output `source_spec` and `target_spec`
            spec_masked = spec[0]
            spec_target = spec[1]
        elif len(spec) == 1:
            spec_masked = spec[0] # (batch_size, seq_len, feat_dim)
            spec_target = copy.deepcopy(spec[0]) # (batch_size, seq_len, feat_dim)
        else:
            raise ValueError

        # Record length for each uttr
        spec_len = (spec_target.sum(dim=-1) != 0).long().sum(dim=-1).tolist()
        batch_size = spec_target.shape[0]
        seq_len = spec_target.shape[1]
        
        pos_enc = fast_position_encoding(seq_len, config['position_encoding_size']) # (seq_len, position_encoding_size)
        mask_label = torch.zeros_like(spec_target, dtype=torch.uint8) \
                     if config['mask_proportion'] != 0 or config['mask_frequency'] != 0 \
                     else torch.ones_like(spec_target, dtype=torch.uint8)
        attn_mask = torch.ones((batch_size, seq_len)) # (batch_size, seq_len)

        for idx in range(batch_size):
            # zero vectors for padding dimension
            attn_mask[idx, spec_len[idx]:] = 0

            def _starts_to_intervals(starts, consecutive):
                tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
                offset = torch.arange(consecutive).expand_as(tiled)
                intervals = tiled + offset
                return intervals.view(-1)
            
            # time masking
            if config['mask_proportion'] > 0:
                mask_consecutive = random.randint(config['mask_consecutive_min'], config['mask_consecutive_max'])
                valid_start_max = max(spec_len[idx] - mask_consecutive - 1, 0) # compute max valid start point for a consecutive mask
                proportion = round(spec_len[idx] * config['mask_proportion'] / mask_consecutive)
                if config['mask_allow_overlap']:
                    # draw `proportion` samples from the range (0, valid_index_range) and without replacement
                    chosen_starts = torch.randperm(valid_start_max + 1)[:proportion]
                else:
                    mask_bucket_size = round(mask_consecutive * config['mask_bucket_ratio'])
                    rand_start = random.randint(0, min(mask_consecutive, valid_start_max))
                    valid_starts = torch.arange(rand_start, valid_start_max + 1, mask_bucket_size)
                    chosen_starts = valid_starts[torch.randperm(len(valid_starts))[:proportion]]
                chosen_intervals = _starts_to_intervals(chosen_starts, mask_consecutive)
                
                # determine whether to mask / random / or do nothing to the frame
                dice = random.random()
                # mask to zero
                if dice < 0.8:
                    spec_masked[idx, chosen_intervals, :] = 0
                # replace to random frames
                elif dice >= 0.8 and dice < 0.9:
                    random_starts = torch.randperm(valid_start_max + 1)[:proportion]
                    random_intervals = _starts_to_intervals(random_starts, mask_consecutive)
                    spec_masked[idx, chosen_intervals, :] = spec_masked[idx, random_intervals, :]
                # do nothing
                else:
                    pass

                # the gradients will be calculated on chosen frames
                mask_label[idx, chosen_intervals, :] = 1

            # frequency masking
            if config['mask_frequency'] > 0:
                max_width = int(spec_masked.shape[2] * config['mask_frequency'])
                rand_bandwidth = random.randint(0, max_width)
                chosen_starts = torch.randperm(spec_masked.shape[2] - rand_bandwidth)[:1]
                chosen_intervals = _starts_to_intervals(chosen_starts, rand_bandwidth)
                spec_masked[idx, :, chosen_intervals] = 0
                
                # the gradients will be calculated on chosen frames
                mask_label[idx, :spec_len[idx], chosen_intervals] = 1   

        if config['noise_proportion'] > 0:
            # noise augmentation
            dice = random.random()
            if dice < config['noise_proportion']:
                noise_sampler = torch.distributions.Normal(0, 0.2)
                spec_masked += noise_sampler.sample(spec_masked.shape).to(device=spec_masked.device)
        
        valid_batchid = mask_label.view(batch_size, -1).sum(dim=-1).nonzero(as_tuple=False).view(-1)
        spec_masked = spec_masked.to(dtype=torch.float32)[valid_batchid]
        pos_enc = pos_enc.to(dtype=torch.float32)
        mask_label = mask_label.to(dtype=torch.bool)[valid_batchid]
        attn_mask = attn_mask.to(dtype=torch.float32)[valid_batchid]
        spec_target = spec_target.to(dtype=torch.float32)[valid_batchid]

    return spec_masked, pos_enc, mask_label, attn_mask, spec_target
