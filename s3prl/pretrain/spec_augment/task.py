# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ task.py ]
#   Synopsis     [ Spec Augment data processing for pre-training the transformer model ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""
# Adaptive_SpecAugment Author: ShampooWang, cornliu

###############
# IMPORTATION #
###############
import copy
import torch
import random
from s3prl.pretrain.mockingjay.task import fast_position_encoding

############
# CONSTANT #
############
MAX_SEQLEN = 3000


def generate_spec_aug_data(spec, config):
    """
    Process training data for the spce augment task:
        `mask_T`: the time mask parameter T described in the SpecAugment paper, 
                  we use default values based on the LD Policy
                  (In paper: T=100)
        `mask_F`: the frequency mask parameter F described in the SpecAugment paper, 
                  we use default values based on the LD Policy
                  (In paper: F=27:D=80*3, where D is acoustic dimension)
        `num_T` : the number of time masks applied (In paper: mT=2)
        `num_F` : the number of frequency masks applied (In paper: mF=2)
        `p` : upper bound ratio (In paper: p=1.0)
    """
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
                     if config['mask_T'] != 0 or config['mask_F'] != 0 \
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
            upper_bound = spec_len[idx] * config['p'] # upper bound on the time mask so that a time mask cannot be wider than p times the number of time steps
            if config['mask_T'] > 0 and config['mask_T'] < upper_bound:
                for _ in range(config['num_T']):
                    rand_consecutive = random.randint(0, config['mask_T'])
                    chosen_start = torch.randperm(spec_masked.shape[1] - rand_consecutive)[:1]
                    chosen_intervals = _starts_to_intervals(chosen_start, rand_consecutive)
                    spec_masked[idx, chosen_intervals, :] = 0

                    # the gradients will be calculated on chosen frames
                    mask_label[idx, chosen_intervals, :] = 1

            # frequency masking
            if config['mask_F'] > 0:
                for _ in range(config['num_F']):
                    rand_bandwidth = random.randint(0, config['mask_F'])
                    chosen_start = torch.randperm(spec_masked.shape[2] - rand_bandwidth)[:1]
                    chosen_intervals = _starts_to_intervals(chosen_start, rand_bandwidth)
                    spec_masked[idx, :, chosen_intervals] = 0
                    
                    # the gradients will be calculated on chosen frames
                    mask_label[idx, :spec_len[idx], chosen_intervals] = 1   
        
        valid_batchid = mask_label.view(batch_size, -1).sum(dim=-1).nonzero(as_tuple=False).view(-1)
        spec_masked = spec_masked.to(dtype=torch.float32)[valid_batchid]
        pos_enc = pos_enc.to(dtype=torch.float32)
        mask_label = mask_label.to(dtype=torch.bool)[valid_batchid]
        attn_mask = attn_mask.to(dtype=torch.float32)[valid_batchid]
        spec_target = spec_target.to(dtype=torch.float32)[valid_batchid]

    return spec_masked, pos_enc, mask_label, attn_mask, spec_target
