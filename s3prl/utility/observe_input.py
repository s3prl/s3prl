# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ observe_input.py ]
#   Synopsis     [ generates visualizations of the model's input ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import copy
import random
import numpy as np
#-------------#
import torch
import torchaudio
#-------------#
import matplotlib
import matplotlib.pyplot as plt
#-------------#
plt.switch_backend('agg')
seed = 679
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


########
# PATH #
########
utt = 'train-clean-100/1594/135914/1594-135914-0032.flac'
libri_dir = '/media/andi611/1TBSSD/LibriSpeech/'
out_dir = './result/visualization/'


def plot_x(x, name='x', xlabel='Frames'):
    x = x.transpose(1, 0)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(x, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel(xlabel)
    plt.ylabel('Channels')
    plt.tight_layout()
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    fig.canvas.draw()
    fig.savefig(os.path.join(out_dir, name + '.png'), bbox_inches='tight', pad_inches = 0)


def starts_to_intervals(starts, consecutive):
    tiled = starts.expand(consecutive, starts.size(0)).T
    offset = torch.arange(consecutive).expand_as(tiled)
    intervals = tiled + offset
    return intervals.view(-1)


########
# MAIN #
########
def main():
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # plot original
    extracter = torch.hub.load('s3prl/s3prl', 'mel')
    wav, _ = torchaudio.load(os.path.join(libri_dir, utt))
    wavs = [wav]
    x = extracter(wavs)[0].squeeze()
    plot_x(x, name='x', xlabel='A) Original fMLLR feature')

    # to torch tensor
    x = torch.FloatTensor(x)
    x_all = copy.deepcopy(x)

    # time masking
    mask_consecutive_min = 7
    mask_consecutive_max = 7
    mask_proportion = 0.15
    mask_allow_overlap = True

    mask_consecutive = random.randint(mask_consecutive_min, mask_consecutive_max)
    valid_start_max = max(x.size(0) - mask_consecutive - 1, 0) # compute max valid start point for a consecutive mask
    proportion = round(x.size(0) * mask_proportion / mask_consecutive)
    if mask_allow_overlap:
        # draw `proportion` samples from the range (0, valid_index_range) and without replacement
        chosen_starts = torch.randperm(valid_start_max + 1)[:proportion]
    else:
        mask_bucket_size = round(mask_consecutive * mask_bucket_ratio)
        rand_start = random.randint(0, min(mask_consecutive, valid_start_max))
        valid_starts = torch.arange(rand_start, valid_start_max + 1, mask_bucket_size)
        chosen_starts = valid_starts[torch.randperm(len(valid_starts))[:proportion]]
    chosen_intervals = starts_to_intervals(chosen_starts, mask_consecutive)
    
    # mask to zero
    x_time_zero = copy.deepcopy(x)
    x_time_zero[chosen_intervals, :] = 0
    x_all[chosen_intervals, :] = 0
    plot_x(x_time_zero.data.cpu().numpy(), name='x_time_zero', xlabel='B) Mask contiguous segments to zero along temporal axis')

    # replace to random frames
    random_starts = torch.randperm(valid_start_max + 1)[:proportion]
    random_intervals = starts_to_intervals(random_starts, mask_consecutive)
    x_time_replace = copy.deepcopy(x)
    x_time_replace[chosen_intervals, :] = x_time_replace[random_intervals, :]
    plot_x(x_time_replace.data.cpu().numpy(), name='x_time_replace', xlabel='C) Replace contiguous segments with random segments')

    # frequency masking
    mask_frequency = 16
    rand_bandwidth = mask_frequency #random.randint(0, mask_frequency)
    chosen_starts = torch.randperm(x.size(1) - rand_bandwidth)[:1]
    chosen_intervals = starts_to_intervals(chosen_starts, rand_bandwidth)
    x_freq = copy.deepcopy(x)
    x_freq[:, chosen_intervals] = 0
    x_all[:, chosen_intervals] = 0
    plot_x(x_freq.data.cpu().numpy(), name='x_freq', xlabel='D) Mask contiguous segments to zero along channel axis')

    # noise augmentation
    noise_sampler = torch.distributions.Normal(0, 0.2)
    x_noise = copy.deepcopy(x)
    x_noise += noise_sampler.sample(x_noise.shape)
    x_all += noise_sampler.sample(x_all.shape)
    plot_x(x_noise.data.cpu().numpy(), name='x_noise', xlabel='E) Apply sampled Gaussian noise to magnitude')

    # time + freq + noise
    plot_x(x_all.data.cpu().numpy(), name='x_all', xlabel='F) Combining the alterations in B), D), and E)')


if __name__ == '__main__':
	main()