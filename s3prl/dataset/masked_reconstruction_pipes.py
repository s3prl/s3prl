from .base import (
    AugmentedDynamicItemDataset,
    DataPipe,
)
from dataclasses import dataclass

import copy
import torch
import random
import numpy as np
from functools import lru_cache
MAX_SEQLEN = 3000


@dataclass
class PrepareTargetFeat(DataPipe):
    use_copy: bool = True
    source_feat_name: str = "source_feat" # tensors in the shape of: (batch_size, seq_len, feat_dim)
    target_feat_name: str = "target_feat" # tensors in the shape of: (batch_size, seq_len, feat_dim)
    
    def prepare_target_feat(self, feat):
        return copy.deepcopy(feat) if self.use_copy else feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        dataset.add_dynamic_item(self.prepare_target_feat, takes=self.source_feat_name, provides=self.target_feat_name)
        return dataset


@dataclass
class MaskedReconstruction(DataPipe):
    mask_args: dict = None
    source_feat_name: str = "source_feat" # tensors in the shape of: (seq_len, feat_dim)
    target_feat_name: str = "target_feat" # tensors in the shape of: (seq_len, feat_dim)
    masked_data_name: str = "masked_data"

    def generate_masked_data(self, source_feat, target_feat):

        with torch.no_grad():

            # Record length for each uttr
            spec_len = (target_feat.sum(dim=-1) != 0).long().sum(dim=-1).tolist()
            seq_len = target_feat.shape[0]
            
            pos_enc = fast_position_encoding(seq_len, self.mask_args["position_encoding_size"]) # (seq_len, position_encoding_size)
            mask_label = torch.zeros_like(target_feat, dtype=torch.uint8) \
                        if self.mask_args["mask_proportion"] != 0 or self.mask_args["mask_frequency"] != 0 \
                        else torch.ones_like(target_feat, dtype=torch.uint8)
            attn_mask = torch.ones(seq_len) # (seq_len)

            # zero vectors for padding dimension
            attn_mask[spec_len:] = 0

            def _starts_to_intervals(starts, consecutive):
                tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
                offset = torch.arange(consecutive).expand_as(tiled)
                intervals = tiled + offset
                return intervals.view(-1)
            
            # time masking
            if self.mask_args["mask_proportion"] > 0:
                mask_consecutive = random.randint(self.mask_args["mask_consecutive_min"], self.mask_args["mask_consecutive_max"])
                valid_start_max = max(spec_len - mask_consecutive - 1, 0) # compute max valid start point for a consecutive mask
                proportion = round(spec_len * self.mask_args["mask_proportion"] / mask_consecutive)
                if self.mask_args["mask_allow_overlap"]:
                    # draw `proportion` samples from the range (0, valid_index_range) and without replacement
                    chosen_starts = torch.randperm(valid_start_max + 1)[:proportion]
                else:
                    mask_bucket_size = round(mask_consecutive * self.mask_args["mask_bucket_ratio"])
                    rand_start = random.randint(0, min(mask_consecutive, valid_start_max))
                    valid_starts = torch.arange(rand_start, valid_start_max + 1, mask_bucket_size)
                    chosen_starts = valid_starts[torch.randperm(len(valid_starts))[:proportion]]
                chosen_intervals = _starts_to_intervals(chosen_starts, mask_consecutive)
                
                # determine whether to mask / random / or do nothing to the frame
                dice = random.random()
                # mask to zero
                if dice < 0.8:
                    source_feat[chosen_intervals, :] = 0
                # replace to random frames
                elif dice >= 0.8 and dice < 0.9:
                    random_starts = torch.randperm(valid_start_max + 1)[:proportion]
                    random_intervals = _starts_to_intervals(random_starts, mask_consecutive)
                    source_feat[chosen_intervals, :] = source_feat[random_intervals, :]
                # do nothing
                else:
                    pass

                # the gradients will be calculated on chosen frames
                mask_label[chosen_intervals, :] = 1

            # frequency masking
            if self.mask_args["mask_frequency"] > 0:
                max_width = int(target_feat.shape[2] * self.mask_args["mask_frequency"])
                rand_bandwidth = random.randint(0, max_width)
                chosen_starts = torch.randperm(source_feat.shape[2] - rand_bandwidth)[:1]
                chosen_intervals = _starts_to_intervals(chosen_starts, rand_bandwidth)
                source_feat[:, chosen_intervals] = 0
                
                # the gradients will be calculated on chosen frames
                mask_label[:spec_len, chosen_intervals] = 1   
            
            source_feat = source_feat.to(dtype=torch.float32)
            pos_enc = pos_enc.to(dtype=torch.float32)
            mask_label = mask_label.to(dtype=torch.bool)
            attn_mask = attn_mask.to(dtype=torch.float32)
            target_feat = target_feat.to(dtype=torch.float32)

        return dict(
            source_feat=source_feat, 
            pos_enc=pos_enc, 
            mask_label=mask_label, 
            attn_mask=attn_mask, 
            target_feat=target_feat,
        )

    def __call__(self, dataset: AugmentedDynamicItemDataset):

        dataset.add_dynamic_item(
            self.generate_masked_data,
            takes=[self.source_feat_name, self.target_feat_name],
            provides=self.masked_data_name,
        )
        return dataset


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
