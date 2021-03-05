# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the speaker diarization dataset ]
#   Source       [ Refactored from https://github.com/hitachi-speech/EEND ]
#   Author       [ Jiatong Shi ]
#   Copyright    [ Copyleft(c), Johns Hopkins University ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import torch
import numpy as np
from itertools import permutations


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


def calc_diarization_error(pred, label, length):
    (batch_size, max_len, num_output) = label.size()
    # mask the padding part
    mask = np.zeros((batch_size, max_len, num_output))
    for i in range(batch_size):
        mask[i, : length[i], :] = 1

    # pred and label have the shape (batch_size, max_len, num_output)
    label_np = label.data.cpu().numpy().astype(int)
    pred_np = (pred.data.cpu().numpy() > 0).astype(int)
    label_np = label_np * mask
    pred_np = pred_np * mask
    length = length.data.cpu().numpy()

    # compute speech activity detection error
    n_ref = np.sum(label_np, axis=2)
    n_sys = np.sum(pred_np, axis=2)
    speech_scored = float(np.sum(n_ref > 0))
    speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
    speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

    # compute speaker diarization error
    speaker_scored = float(np.sum(n_ref))
    speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
    speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
    n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
    speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
    correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
    num_frames = np.sum(length)
    return (
        correct,
        num_frames,
        speech_scored,
        speech_miss,
        speech_falarm,
        speaker_scored,
        speaker_miss,
        speaker_falarm,
        speaker_error,
    )
