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
import pdb
import IPython
from copy import deepcopy 
import time 
from functools import lru_cache

############
# CONSTANT #
############
DR = 3
HIDDEN_SIZE = 768
MASK_PROPORTION = 0.15
MASK_CONSECUTIVE = 1


def down_sample_frames(spec, dr):
    left_over = spec.shape[1] % dr
    if left_over != 0: spec = spec[:, :-left_over, :]
    spec_stacked = spec.view(spec.shape[0], spec.shape[1]//dr, spec.shape[2]*dr)
    return spec_stacked


def cal_angle(position, hid_idx,hidden_size):
    return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)

def get_posi_angle_vec(position,hidden_size):
    return [cal_angle(position, hid_j,hidden_size) for hid_j in range(hidden_size)]

@lru_cache(maxsize=5)
def static_position_table_f(hidden_size,max_length=2000):

    sinusoid_table          = np.array([get_posi_angle_vec(pos_i,hidden_size) for pos_i in range(2000)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    sinusoid_table          = torch.FloatTensor(sinusoid_table).to(dtype=torch.float32)

    return sinusoid_table

@lru_cache(maxsize=5)
def position_encoding(hidden_size, sinusoid_table, batch_size=None, padding_idx=None):
    ''' Sinusoid position encoding table '''

    if batch_size is not None:
        batch_sinusoid_table =sinusoid_table.expand(batch_size,sinusoid_table.size(0),sinusoid_table.size(1))
        return batch_sinusoid_table # (batch_size, seq_len, hidden_size)
    else:
        return sinusoid_table  # (seq_len, hidden_size)


def process_train_MAM_data(spec, config=None):
    """Process training data for the masked acoustic model"""
    dr = config['downsample_rate'] if config is not None else DR
    hidden_size = config['hidden_size'] if config is not None else HIDDEN_SIZE
    mask_proportion = config['mask_proportion'] if config is not None else MASK_PROPORTION
    mask_consecutive = config['mask_consecutive'] if config is not None else MASK_CONSECUTIVE
    mini_bucket_num = config["mini_bucket_num"]
    consecutive_offset = config["consecutive_offset"]
    temp = []
    
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
        spec_len = np.array([int(sl) for sl in spec_len])

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]
        # start = time.time()
        position_table = static_position_table_f(hidden_size)[:seq_len]
        pos_enc = position_encoding(hidden_size, position_table, batch_size) # (batch_size, seq_len, hidden_size)
        # end = time.time()

        # with open(f"cache_position_embedding.txt","a+") as d: 
        #     d.write(f"cache time is {end - start:7.6f}\n")

        
        mask_label = np.zeros_like(spec_stacked)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)
        batch_consecutives                         = np.random.choice(np.arange(0,mask_consecutive), size=batch_size) +1
        batch_random_dices                         = torch.rand(batch_size)
        batch_valid_indexes                        = spec_len - batch_consecutives - 1
        batch_proportions                          = spec_len * mask_proportion // batch_consecutives
        batch_proportions[batch_proportions == 0 ] = 1
        batch_start_points                         = torch.randint(low=0, high=mask_consecutive, size=(batch_size,)).data.cpu().numpy()
        batch_buckets_num                          = (batch_valid_indexes - batch_start_points) // (batch_consecutives + consecutive_offset)
        # end_batch_time = time.time()

        # with open(f"batch_time.txt","a+") as d:
        #     d.write(f"batch time is {end_batch_time - start_two_time:7.6f}\n")
        
        for idx in range(len(spec_stacked)):
            
            # determine whether to mask / random / or do nothing to the frame
            if batch_buckets_num[idx] < mini_bucket_num:
                temp += [idx]
                continue 
            step = (batch_consecutives[idx] + consecutive_offset)
            bound_indexes = np.arange(batch_start_points[idx], batch_valid_indexes[idx], step ) 
            chosen_index = torch.LongTensor(np.random.permutation(bound_indexes)[:int(batch_proportions[idx])]) # draw `proportion` samples from the range (0, valid_index_range) and without replacement
            
            chosen_index     = chosen_index.unsqueeze(-1)
            mapping          = chosen_index.expand(chosen_index.size(0),batch_consecutives[idx])
            offset           = torch.LongTensor(np.arange(batch_consecutives[idx])).expand(chosen_index.size(0), batch_consecutives[idx])
            indexes          = mapping + offset
            one_line_indexes = indexes.view(1,-1) 
            # mask to zero
            if bool(batch_random_dices[idx] < 0.8):
                spec_masked[idx][one_line_indexes] = 0
            # replace to random frames
            elif bool(batch_random_dices[idx] >= 0.8) and bool(batch_random_dices[idx] < 0.9):
                length = batch_consecutives[idx]*(chosen_index.shape[0])
                random_index = np.random.permutation(batch_valid_indexes[idx])[:length]
                spec_masked[idx][one_line_indexes] = spec_masked[idx][random_index]
            # do nothing
            else:
                pass

            # the gradients will be calculated on all chosen frames
            mask_label[idx][one_line_indexes] = 1

            # zero vectors for padding dimension
            attn_mask[idx][spec_len[idx]:] = 0

        # end_time =time.time()

        # with open(f"loop_time.txt","a+") as d:
        #     d.write(f"loop time is {end_time - end_batch_time:7.6f}\n")

        # with open(f"loop_all_time.txt","a+") as d:
        #     d.write(f"loop time is {end_time - start_two_time:7.6f}\n")

        spec_masked = spec_masked.to(dtype=torch.float32)
        mask_label = torch.ByteTensor(mask_label).to(dtype=torch.bool)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)
        spec_stacked = spec_stacked.to(dtype=torch.float32)
        # if len(temp) != 0:
        #     print(f"\n miss {len(temp)} sampe data in batch {len(spec_masked)} \n")

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
        position_table = static_position_table_f(hidden_size)[:seq_len]
        pos_enc = position_encoding(hidden_size,position_table, batch_size) # (batch_size, seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(spec_stacked)):
            pos_enc[idx][spec_len[idx]:] = 0  
            attn_mask[idx][spec_len[idx]:] = 0

        spec_stacked = spec_stacked.to(dtype=torch.float32)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)

    return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)