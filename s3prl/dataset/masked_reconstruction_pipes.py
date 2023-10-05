import copy
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .base import AugmentedDynamicItemDataset, DataPipe

MAX_SEQLEN = 10000


@dataclass
class PrepareTargetFeat(DataPipe):
    use_copy: bool = True
    source_feat_name: str = "source_feat"
    target_feat_name: str = "target_feat"

    def prepare_target_feat(self, feat):
        target_feat = copy.deepcopy(feat) if self.use_copy else feat
        return target_feat.to(dtype=torch.float32)

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        dataset.add_dynamic_item(
            self.prepare_target_feat,
            takes=self.source_feat_name,
            provides=self.target_feat_name,
        )
        return dataset


@dataclass
class MaskedReconstruction(DataPipe):
    position_encoding_size: int = 768
    mask_proportion: float = 0.15
    mask_consecutive_min: int = 7
    mask_consecutive_max: int = 7
    mask_allow_overlap: bool = True
    mask_bucket_ratio: float = 1.5
    mask_frequency: int = 0
    source_feat_name: str = "source_feat"
    target_feat_name: str = "target_feat"
    masked_feat_name: str = "masked_feat"
    pos_enc_name: str = "pos_enc"
    attn_mask_name: str = "attn_mask"
    label_mask_name: str = "label_mask"
    """
    Args:
        position_encoding_size (int): this should be identical to `hidden_size`
        mask_proportion (float): mask this percentage of all spectrogram frames in each sequence at random during MAM training
        mask_consecutive_min (int): mask this amount of consecutive frames
        mask_consecutive_max (int): mask this amount of consecutive frames
        mask_allow_overlap (bool): allow overlap masking
        mask_bucket_ratio (float): only used when overlap is not allowed. sample a mask from each bucket in size of [sampled mask_consecutive * mask_bucket_ratio]
        mask_frequency (float): mask maximum this percentage of frequency bands, set to 0 for no frequency mask
        source_feat_name (str): handle for the `takes` (input)
        target_feat_name (str): handle for the `takes` (input)
        masked_feat_name (str): handle for the `provides` (output)
        pos_enc_name (str): handle for the `provides` (output)
        attn_mask_name (str): handle for the `provides` (output)
        label_mask_name (str): handle for the `provides` (output)
    """

    def generate_masked_data(self, source_feat, target_feat):

        with torch.no_grad():

            masked_feat = copy.deepcopy(source_feat)

            # Record length for each uttr
            spec_len = (target_feat.sum(dim=-1) != 0).long().sum(dim=-1).tolist()
            seq_len = target_feat.shape[0]

            pos_enc = fast_position_encoding(
                seq_len, self.position_encoding_size
            )  # (seq_len, position_encoding_size)
            label_mask = (
                torch.zeros_like(target_feat, dtype=torch.uint8)
                if self.mask_proportion != 0 or self.mask_frequency != 0
                else torch.ones_like(target_feat, dtype=torch.uint8)
            )
            attn_mask = torch.ones(seq_len)  # (seq_len)

            # zero vectors for padding dimension
            attn_mask[spec_len:] = 0

            def _starts_to_intervals(starts, consecutive):
                tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
                offset = torch.arange(consecutive).expand_as(tiled)
                intervals = tiled + offset
                return intervals.view(-1)

            # time masking
            if self.mask_proportion > 0:
                mask_consecutive = random.randint(
                    self.mask_consecutive_min,
                    self.mask_consecutive_max,
                )
                valid_start_max = max(
                    spec_len - mask_consecutive - 1, 0
                )  # compute max valid start point for a consecutive mask
                proportion = round(spec_len * self.mask_proportion / mask_consecutive)
                if self.mask_allow_overlap:
                    # draw `proportion` samples from the range (0, valid_index_range) and without replacement
                    chosen_starts = torch.randperm(valid_start_max + 1)[:proportion]
                else:
                    mask_bucket_size = round(mask_consecutive * self.mask_bucket_ratio)
                    rand_start = random.randint(
                        0, min(mask_consecutive, valid_start_max)
                    )
                    valid_starts = torch.arange(
                        rand_start, valid_start_max + 1, mask_bucket_size
                    )
                    chosen_starts = valid_starts[
                        torch.randperm(len(valid_starts))[:proportion]
                    ]
                chosen_intervals = _starts_to_intervals(chosen_starts, mask_consecutive)

                # determine whether to mask / random / or do nothing to the frame
                dice = random.random()
                # mask to zero
                if dice < 0.8:
                    masked_feat[chosen_intervals, :] = 0
                # replace to random frames
                elif dice >= 0.8 and dice < 0.9:
                    random_starts = torch.randperm(valid_start_max + 1)[:proportion]
                    random_intervals = _starts_to_intervals(
                        random_starts, mask_consecutive
                    )
                    masked_feat[chosen_intervals, :] = masked_feat[random_intervals, :]
                # do nothing
                else:
                    pass

                # the gradients will be calculated on chosen frames
                label_mask[chosen_intervals, :] = 1

            # frequency masking
            if self.mask_frequency > 0:
                max_width = int(masked_feat.shape[-1] * self.mask_frequency)
                rand_bandwidth = random.randint(0, max_width)
                chosen_starts = torch.randperm(masked_feat.shape[-1] - rand_bandwidth)[
                    :1
                ]
                chosen_intervals = _starts_to_intervals(chosen_starts, rand_bandwidth)
                masked_feat[:, chosen_intervals] = 0

                # the gradients will be calculated on chosen frames
                label_mask[:spec_len, chosen_intervals] = 1

            masked_feat = masked_feat.to(dtype=torch.float32)
            pos_enc = pos_enc.to(dtype=torch.float32)
            attn_mask = attn_mask.to(dtype=torch.float32)
            label_mask = label_mask.to(dtype=torch.bool)

        return masked_feat, pos_enc, attn_mask, label_mask

    def __call__(self, dataset: AugmentedDynamicItemDataset):

        dataset.add_dynamic_item(
            self.generate_masked_data,
            takes=[self.source_feat_name, self.target_feat_name],
            provides=[
                self.masked_feat_name,
                self.pos_enc_name,
                self.attn_mask_name,
                self.label_mask_name,
            ],
        )
        return dataset


@lru_cache(maxsize=128)
def get_sinusoid_table(hidden_size):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(hidden_size)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(MAX_SEQLEN)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def fast_position_encoding(seq_len, hidden_size, batch_size=None, padding_idx=None):

    assert (
        seq_len <= MAX_SEQLEN
    ), f"constant MAX_SEQLEN ({MAX_SEQLEN}) in mam.py < received seq_len ({seq_len})"
    table = get_sinusoid_table(hidden_size)[:seq_len]

    if padding_idx is not None:
        # deepcopy will slow down whole process when positional table is too large
        # this path is dreprecated and should never be used
        table = copy.deepcopy(table)
        table[padding_idx:] = 0.0  # zero vector for padding dimension

    if batch_size is not None:
        # using expand will not cause extra CPU memory allocation issue
        # however, the expanded tensor after put into GPU still need
        # GPU memory of expanded size, which should be avoided when
        # positional table is large
        # this path is not recommended
        batch_table = table.expand(batch_size, -1, -1)
        return batch_table  # (batch_size, seq_len, hidden_size)
    else:
        # this path is most recommended, no extra CPU and GPU memory allocation
        # after getting the (seq_len, hidden_size) tensor, one should first put
        # this tensor into GPU then expand it
        return table  # (seq_len, hidden_size)
