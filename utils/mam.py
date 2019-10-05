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


# TODO: load this from yaml
dr = 3
hidden_size = 1024
mask_proportion = 0.15


def down_sample_frames(spec):
    left_over = spec.shape[1] % dr
    if left_over != 0: spec = spec[:, :-left_over, :]
    spec_stacked = spec.view(spec.shape[0], spec.shape[1]//dr, spec.shape[2]*dr)
    return spec_stacked


def position_encoding(seq_len, batch_size=None, padding_idx=None):
    ''' Sinusoid position encoding table '''
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(hidden_size)]

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


def process_train_MAM_data(spec):
    """Process training data for the masked acoustic model"""
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
        spec_masked = down_sample_frames(source_spec) # (batch_size, seq_len, mel_dim * dr)
        spec_stacked = down_sample_frames(target_spec) # (batch_size, seq_len, mel_dim * dr)
        assert(spec_masked.shape[1] == spec_stacked.shape[1]), 'Input and output spectrogram should have the same shape'

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = position_encoding(seq_len, batch_size) # (batch_size, seq_len, hidden_size)
        mask_label = np.zeros_like(spec_stacked)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        for idx in range(len(spec_stacked)):
            
            chose_proportion = int(spec_len[idx]*mask_proportion) # chooses % of the frame positions at random for prediction
            sub_mask_proportion = int(chose_proportion*0.8) # replace the i-th frame with (1) the [MASK] frame 80% of the time
            sub_rand_proportion = int(chose_proportion*0.1) # a random frame 10% of the time
            
            sample_index = random.sample(range(spec_len[idx]), chose_proportion + sub_rand_proportion) # sample the chosen_index and random frames
            chosen_index = sample_index[:chose_proportion]
            masked_index = chosen_index[:sub_mask_proportion]

            if sub_rand_proportion > 0:
                random_index = chosen_index[-sub_rand_proportion:]
                random_frames = sample_index[-sub_rand_proportion:]
                spec_masked[idx][random_index] = spec_masked[idx][random_frames]
            
            spec_masked[idx][masked_index] = 0 # mask frames to zero
            mask_label[idx][chosen_index] = 1 # the frames where gradients will be calculated on 

            # zero vectors for padding dimension
            pos_enc[idx][spec_len[idx]:] = 0  
            attn_mask[idx][spec_len[idx]:] = 0

        spec_masked = spec_masked.to(dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(dtype=torch.float32)
        mask_label = torch.ByteTensor(mask_label).to(dtype=torch.uint8)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)
        spec_stacked = spec_stacked.to(dtype=torch.float32)

    return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked


def process_test_MAM_data(spec):
    """Process testing data for the masked acoustic model"""
    
    with torch.no_grad():
        if len(spec) != 1:
            raise NotImplementedError('Input spec sould be a tuple of: (spec,), where `spec` has shape BxTxD.')

        # Down sample
        spec_stacked = down_sample_frames(spec[0]) # (batch_size, seq_len, mel_dim * dr)

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = position_encoding(seq_len, batch_size) # (batch_size, seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(spec_stacked)):
            pos_enc[idx][spec_len[idx]:] = 0  
            attn_mask[idx][spec_len[idx]:] = 0 

        spec_stacked = spec_stacked.to(dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(dtype=torch.float32)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)

    return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)


def process_train_MAM_data_spectrum_order(spec):
    """Process training data for the masked acoustic model"""
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
        spec_masked = down_sample_frames(source_spec) # (batch_size, seq_len, mel_dim * dr)
        spec_stacked = down_sample_frames(target_spec) # (batch_size, seq_len, mel_dim * dr)
        assert(spec_masked.shape[1] == spec_stacked.shape[1]), 'Input and output spectrogram should have the same shape'

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = position_encoding(seq_len, batch_size) # (batch_size, seq_len, hidden_size)
        mask_label = np.zeros_like(spec_stacked)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        order_label = torch.zeros(batch_size,2)
        

        for idx in range(len(spec_stacked)):

            #ignore middle 20 % samples and divide the spec to two part to the second task spec order prediction
            sample_length = int(spec_len[idx])
            first_proportion_input  = spec_masked[idx][: int(0.4*sample_length)]
            second_proportion_input = spec_masked[idx][int(0.6*sample_length): ]
            ignore_part_input = spec_masked[idx][int(0.4*sample_length): int(0.6*sample_length) ]

            first_proportion_output  = spec_stacked[idx][: int(0.4*sample_length)]
            second_proportion_output = spec_stacked[idx][int(0.6*sample_length): ]
            ignore_part_output = spec_stacked[idx][int(0.4*sample_length): int(0.6*sample_length) ]

            ignore_indexes = list(range(int(0.4*sample_length),int(0.6*sample_length)))

            if random.uniform(0, 1) > 0.5:
                order_label[idx][1] = 1
                spec_masked[idx] = torch.cat([first_proportion_input,ignore_part_input,second_proportion_input],dim=0)
                spec_stacked[idx] = torch.cat([first_proportion_output,ignore_part_output,second_proportion_output],dim=0)
            else:
                order_label[idx][0] = 1
                spec_masked[idx] = torch.cat([second_proportion_input,ignore_part_input,first_proportion_input],dim=0)
                spec_stacked[idx] = torch.cat([first_proportion_output,ignore_part_output,second_proportion_output],dim=0)

            chose_proportion = int(spec_len[idx]*mask_proportion) # chooses % of the frame positions at random for prediction
            sub_mask_proportion = int(chose_proportion*0.8) # replace the i-th frame with (1) the [MASK] frame 80% of the time
            sub_rand_proportion = int(chose_proportion*0.1) # a random frame 10% of the time

            all_range = list(range(spec_len[idx]))
            just_two_proportion = [x for x in all_range if x not in ignore_indexes] #we will keep ignore part data to clean
            sample_index = random.sample(just_two_proportion, chose_proportion + sub_rand_proportion) # sample the chosen_index and random frames
            chosen_index = sample_index[:chose_proportion]
            masked_index = chosen_index[:sub_mask_proportion]

            if sub_rand_proportion > 0:
                random_index = chosen_index[-sub_rand_proportion:]
                random_frames = sample_index[-sub_rand_proportion:]
                spec_masked[idx][random_index] = spec_masked[idx][random_frames]
            
            spec_masked[idx][masked_index] = 0 # mask frames to zero
            mask_label[idx][chosen_index] = 1 # the frames where gradients will be calculated on 

            # zero vectors for padding dimension
            pos_enc[idx][spec_len[idx]:] = 0  
            attn_mask[idx][spec_len[idx]:] = 0

        spec_masked = spec_masked.to(dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(dtype=torch.float32)
        mask_label = torch.ByteTensor(mask_label).to(dtype=torch.uint8)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)
        spec_stacked = spec_stacked.to(dtype=torch.float32)

    return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked, order_label




def process_train_MAM_data_spectrum_order_random_mask(spec):
    """Process training data for the masked acoustic model"""
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
        spec_masked = down_sample_frames(source_spec) # (batch_size, seq_len, mel_dim * dr)
        spec_stacked = down_sample_frames(target_spec) # (batch_size, seq_len, mel_dim * dr)
        assert(spec_masked.shape[1] == spec_stacked.shape[1]), 'Input and output spectrogram should have the same shape'
        
        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = position_encoding(seq_len, batch_size) # (batch_size, seq_len, hidden_size)
        mask_label = np.zeros_like(spec_stacked)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        order_label = torch.zeros(batch_size,2)
        

        for idx in range(len(spec_stacked)):

            #ignore middle 20 % samples and divide the spec to two part to the second task spec order prediction
            sample_length = int(spec_len[idx])
            first_proportion_input  = spec_masked[idx][: int(0.4*sample_length)]
            second_proportion_input = spec_masked[idx][int(0.6*sample_length): ]
            ignore_part_input = spec_masked[idx][int(0.4*sample_length): int(0.6*sample_length) ]

            first_proportion_output  = spec_stacked[idx][: int(0.4*sample_length)]
            second_proportion_output = spec_stacked[idx][int(0.6*sample_length): ]
            ignore_part_output = spec_stacked[idx][int(0.4*sample_length): int(0.6*sample_length) ]

            ignore_indexes = list(range(int(0.4*sample_length),int(0.6*sample_length)))

            if random.uniform(0, 1) > 0.5:
                order_label[idx][1] = 1
                spec_masked[idx] = torch.cat([first_proportion_input,ignore_part_input,second_proportion_input],dim=0)
                spec_stacked[idx] = torch.cat([first_proportion_output,ignore_part_output,second_proportion_output],dim=0)
            else:
                order_label[idx][0] = 1
                spec_masked[idx] = torch.cat([second_proportion_input,ignore_part_input,first_proportion_input],dim=0)
                spec_stacked[idx] = torch.cat([first_proportion_output,ignore_part_output,second_proportion_output],dim=0)

            chose_proportion = int(spec_len[idx]*mask_proportion) # chooses % of the frame positions at random for prediction
            tose =random.uniform(0,1)
            target = None # initialize
            if tose < 0.8:
                mask_proportion = int(chose_proportion) # replace the i-th frame with (1) the [MASK] frame 80% of the time
                target = "mask"
            elif tose > 0.9:
                random_proportion = int(chose_proportion) # a random frame 10% of the time
                target = "random"
            else:
                target = "nochange"
                
            all_range = list(range(spec_len[idx]))
            just_two_proportion = [x for x in all_range if x not in ignore_indexes] #we will keep ignore part data to clean
            
            if target == "mask":
                masked_index = random.sample(just_two_proportion, chose_proportion) # sample the chosen_index and random frames
                spec_masked[idx][masked_index] = 0 # mask frames to zero
                mask_label[idx][masked_index] = 1 # the frames where gradients will be calculated on 

            if target == "random":
                random_index = random.sample(just_two_proportion, chose_proportion)
                random_frames_index = random.sample(just_two_proportion, chose_proportion)
                spec_masked[idx][random_index] = spec_masked[idx][random_frames]
                mask_label[idx][masked_index] = 1 # the frames where gradients will be calculated on 
            
            if target == "nochage":
                choose_index = random.sample(just_two_proportion, chose_proportion)
                mask_label[idx][masked_index] = 1 # the frames where gradients will be calculated on 

            # zero vectors for padding dimension
            pos_enc[idx][spec_len[idx]:] = 0  
            attn_mask[idx][spec_len[idx]:] = 0

        spec_masked = spec_masked.to(dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(dtype=torch.float32)
        mask_label = torch.ByteTensor(mask_label).to(dtype=torch.uint8)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)
        spec_stacked = spec_stacked.to(dtype=torch.float32)
        order_label = order_label.to(dtype=torch.int32)

    return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked, order_label
