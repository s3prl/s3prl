"""
Permutation Invariant Training (PIT) loss

Authors:
  * Jiatong Shi 2021
"""

from itertools import permutations

import numpy as np
import torch

__all__ = [
    "pit_loss",
]


# compute mask to remove the padding positions
def create_length_mask(length, max_len, num_output, device):
    batch_size = len(length)
    mask = torch.zeros(batch_size, max_len, num_output)
    for i in range(batch_size):
        mask[i, : length[i], :] = 1
    mask = mask.to(device)
    return mask


# compute loss for a single permutation
def pit_loss_single_permute(output, label, length):
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    mask = create_length_mask(length, label.size(1), label.size(2), label.device)
    loss = bce_loss(output, label)
    loss = loss * mask
    loss = torch.sum(torch.mean(loss, dim=2), dim=1)
    loss = torch.unsqueeze(loss, dim=1)
    return loss


def pit_loss(output, label, length):
    """
    The Permutation Invariant Training loss

    Args:
        output (torch.FloatTensor): prediction in (batch_size, seq_len, num_class)
        label (torch.FloatTensor): label in the same shape as :code:`output`
        length (torch.LongTensor): the valid length of each instance. :code:`output` and :code:`label`
            share the same valid length

    Returns:
        tuple:

        1. loss (torch.FloatTensor)
        2. min_idx (int): the id with the minimum loss
        3. all the permutation
    """
    num_output = label.size(2)
    device = label.device
    permute_list = [np.array(p) for p in permutations(range(num_output))]
    loss_list = []
    for p in permute_list:
        label_perm = label[:, :, p]
        loss_perm = pit_loss_single_permute(output, label_perm, length)
        loss_list.append(loss_perm)
    loss = torch.cat(loss_list, dim=1)
    min_loss, min_idx = torch.min(loss, dim=1)
    loss = torch.sum(min_loss) / torch.sum(length.float().to(device))
    return loss, min_idx, permute_list


def get_label_perm(label, perm_idx, perm_list):
    batch_size = len(perm_idx)
    label_list = []
    for i in range(batch_size):
        label_list.append(label[i, :, perm_list[perm_idx[i]]].data.cpu().numpy())
    return torch.from_numpy(np.array(label_list)).float()
