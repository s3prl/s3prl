# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ transformer/mam.py ]
#   Synopsis     [ Moasked Acoustic Model data processing for pre-training the transformer model ]
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
MASK_BUCKET_RATIO = 1.2
MASK_FREQUENCY = 8
NOISE_PROPORTION = 0.15
MAX_SEQLEN = 5000


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


def process_train_MAM_data(spec, config=None):
    """Process training data for the masked acoustic model"""

    dr = config['downsample_rate'] if config is not None else DR
    hidden_size = config['hidden_size'] if config is not None else HIDDEN_SIZE
    mask_proportion = config['mask_proportion'] if config is not None else MASK_PROPORTION
    mask_consecutive_min = config['mask_consecutive_min'] if config is not None else MASK_CONSECUTIVE
    mask_consecutive_max = config['mask_consecutive_max'] if config is not None else MASK_CONSECUTIVE
    mask_allow_overlap = config['mask_allow_overlap'] if config is not None else True
    mask_bucket_ratio = config['mask_bucket_ratio'] if config is not None else MASK_BUCKET_RATIO
    mask_frequency = config['mask_frequency'] if config is not None else MASK_FREQUENCY
    noise_proportion = config['noise_proportion'] if config is not None else NOISE_PROPORTION
    test_reconstruct = False

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
        spec_len = (spec_stacked.sum(dim=-1) != 0).long().sum(dim=-1).tolist()
        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]
        
        pos_enc = fast_position_encoding(seq_len, hidden_size) # (seq_len, hidden_size)
        mask_label = torch.zeros_like(spec_stacked, dtype=torch.uint8)
        attn_mask = torch.ones((batch_size, seq_len)) # (batch_size, seq_len)

        for idx in range(batch_size):
            # zero vectors for padding dimension
            attn_mask[idx, spec_len[idx]:] = 0

            if test_reconstruct:
                mask_label[idx, :, :] = 1
                continue

            def starts_to_intervals(starts, consecutive):
                tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
                offset = torch.arange(consecutive).expand_as(tiled)
                intervals = tiled + offset
                return intervals.view(-1)
            
            # time masking
            mask_consecutive = random.randint(mask_consecutive_min, mask_consecutive_max)
            valid_start_max = max(spec_len[idx] - mask_consecutive - 1, 0) # compute max valid start point for a consecutive mask
            proportion = round(spec_len[idx] * mask_proportion / mask_consecutive)
            if mask_allow_overlap:
                # draw `proportion` samples from the range (0, valid_index_range) and without replacement
                chosen_starts = torch.randperm(valid_start_max + 1)[:proportion]
            else:
                mask_bucket_size = round(mask_consecutive * mask_bucket_ratio)
                rand_start = random.randint(0, min(mask_consecutive, valid_start_max))
                valid_starts = torch.arange(rand_start, valid_start_max + 1, mask_bucket_size)
                chosen_starts = valid_starts[torch.randperm(len(valid_starts))[:proportion]]
            chosen_intervals = starts_to_intervals(chosen_starts, mask_consecutive)
            
            # determine whether to mask / random / or do nothing to the frame
            dice = random.random()
            # mask to zero
            if dice < 0.8:
                spec_masked[idx, chosen_intervals, :] = 0
            # replace to random frames
            elif dice >= 0.8 and dice < 0.9:
                random_starts = torch.randperm(valid_start_max + 1)[:proportion]
                random_intervals = starts_to_intervals(random_starts, mask_consecutive)
                spec_masked[idx, chosen_intervals, :] = spec_masked[idx, random_intervals, :]
            # do nothing
            else:
                pass

            # the gradients will be calculated on chosen frames
            mask_label[idx, chosen_intervals, :] = 1

            # frequency masking
            if mask_frequency > 0:
                rand_bandwidth = random.randint(0, mask_frequency)
                chosen_starts = torch.randperm(spec_masked.shape[2] - rand_bandwidth)[:1]
                chosen_intervals = starts_to_intervals(chosen_starts, rand_bandwidth)
                spec_masked[idx, :, chosen_intervals] = 0
                
                # the gradients will be calculated on chosen frames
                mask_label[idx, :, chosen_intervals] = 1   

        if not test_reconstruct:
            # noise augmentation
            dice = random.random()
            if dice < noise_proportion:
                noise_sampler = torch.distributions.Normal(0, 0.2)
                spec_masked += noise_sampler.sample(spec_masked.shape).to(device=spec_masked.device)
        
        valid_batchid = mask_label.view(batch_size, -1).sum(dim=-1).nonzero().view(-1)
        batch_is_valid = len(valid_batchid) > 0
        spec_masked = spec_masked.to(dtype=torch.float32)[valid_batchid]
        pos_enc = pos_enc.to(dtype=torch.float32)
        mask_label = mask_label.to(dtype=torch.bool)[valid_batchid]
        attn_mask = attn_mask.to(dtype=torch.float32)[valid_batchid]
        spec_stacked = spec_stacked.to(dtype=torch.float32)[valid_batchid]

    return batch_is_valid, spec_masked, pos_enc, mask_label, attn_mask, spec_stacked


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
        spec_len = (spec_stacked.sum(dim=-1) != 0).long().sum(dim=-1).tolist()
        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = fast_position_encoding(seq_len, hidden_size) # (seq_len, hidden_size)
        attn_mask = torch.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(spec_stacked)):
            attn_mask[idx, spec_len[idx]:] = 0 

        spec_stacked = spec_stacked.to(dtype=torch.float32)
        pos_enc = pos_enc.to(dtype=torch.float32)
        attn_mask = attn_mask.to(dtype=torch.float32)

    return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)
