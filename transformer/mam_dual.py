# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ transformer/mam_dual.py ]
#   Synopsis     [ Masked Acoustic Model data processing for pre-training the dual transformer model ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import copy
import torch
import random
from transformer.mam import down_sample_frames, fast_position_encoding


############
# CONSTANT #
############
DR = 1
HIDDEN_SIZE = 768
MASK_PROPORTION = 0.15
MASK_CONSECUTIVE = 7
MASK_BUCKET_RATIO = 1.2
MASK_FREQUENCY = 8
NOISE_PROPORTION = 0.15
MAX_SEQLEN = 5000


def process_dual_train_MAM_data(spec, config=None):
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

        # create dual input
        time_masked = spec_masked
        if mask_proportion == 0 and mask_frequency == 0:
            freq_masked = spec_masked
        else:
            freq_masked = copy.deepcopy(spec_masked)

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
            if mask_proportion > 0:
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
                    time_masked[idx, chosen_intervals, :] = 0
                # replace to random frames
                elif dice >= 0.8 and dice < 0.9:
                    random_starts = torch.randperm(valid_start_max + 1)[:proportion]
                    random_intervals = starts_to_intervals(random_starts, mask_consecutive)
                    time_masked[idx, chosen_intervals, :] = time_masked[idx, random_intervals, :]
                # do nothing
                else:
                    pass

                # the gradients will be calculated on chosen frames
                mask_label[idx, chosen_intervals, :] = 1

            # frequency masking
            if mask_frequency > 0:
                rand_bandwidth = random.randint(0, mask_frequency)
                chosen_starts = torch.randperm(freq_masked.shape[2] - rand_bandwidth)[:1]
                chosen_intervals = starts_to_intervals(chosen_starts, rand_bandwidth)
                freq_masked[idx, :, chosen_intervals] = 0
                
                # the gradients will be calculated on chosen frames
                mask_label[idx, :, chosen_intervals] = 1   

        if not test_reconstruct and noise_proportion > 0:
            # noise augmentation
            dice = random.random()
            noise_sampler = torch.distributions.Normal(0, 0.2)
            if dice < noise_proportion:
                time_masked += noise_sampler.sample(time_masked.shape).to(device=time_masked.device)
            dice = random.random()
            if dice < noise_proportion:
                freq_masked += noise_sampler.sample(freq_masked.shape).to(device=freq_masked.device)

        
        valid_batchid = mask_label.view(batch_size, -1).sum(dim=-1).nonzero().view(-1)
        batch_is_valid = len(valid_batchid) > 0
        time_masked = time_masked.to(dtype=torch.float32)[valid_batchid]
        freq_masked = freq_masked.to(dtype=torch.float32)[valid_batchid]
        pos_enc = pos_enc.to(dtype=torch.float32)
        mask_label = mask_label.to(dtype=torch.bool)[valid_batchid]
        attn_mask = attn_mask.to(dtype=torch.float32)[valid_batchid]
        spec_stacked = spec_stacked.to(dtype=torch.float32)[valid_batchid]

    return batch_is_valid, time_masked, freq_masked, pos_enc, mask_label, attn_mask, spec_stacked