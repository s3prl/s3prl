import torch
import torch.nn as nn
import torch.nn.functional as F


def get_length_masks(lengths):
    positions = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.size(0), -1).to(lengths.device)
    length_masks = torch.lt(positions, lengths.unsqueeze(-1))
    return length_masks


def mask_mean(tensors, length_masks):
    summations = (tensors * length_masks).sum(dim=1, keepdim=True)
    means = summations / length_masks.sum(dim=1, keepdim=True)
    assert torch.isnan(means).sum() == 0 and torch.isinf(means).sum() == 0
    return means


def mask_std(tensors, length_masks, means=None):
    if means is None:
        means = mask_mean(tensors, length_masks)

    diffs = (tensors - means).pow(2)
    diff_sums = (diffs * length_masks).sum(dim=1, keepdim=True)
    stds = (diff_sums / (length_masks.sum(dim=1, keepdim=True) - 1)).pow(0.5)
    assert torch.isnan(stds).sum() == 0 and torch.isinf(stds).sum() == 0
    return stds


def mask_normalize(tensors, length_masks):
    means = mask_mean(tensors, length_masks)
    stds = mask_std(tensors, length_masks, means)
    normalized_tensors = (tensors - means) / (stds + 1e-8)
    assert torch.isnan(normalized_tensors).sum() == 0 and torch.isinf(normalized_tensors).sum() == 0
    return normalized_tensors


if __name__ == '__main__':
    from ipdb import set_trace

    inputs = torch.randn(10, 50, 100)
    length_masks = get_length_masks(torch.ones(10) * 50).unsqueeze(-1)

    mask_means = mask_mean(inputs, length_masks)
    mask_stds = mask_std(inputs, length_masks, mask_means)

    assert torch.allclose(mask_means, inputs.mean(dim=1, keepdim=True))
    assert torch.allclose(mask_stds, inputs.std(dim=1, keepdim=True))
