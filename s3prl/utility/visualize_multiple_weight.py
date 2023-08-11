# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utils/visualize_weight.py ]
#   Synopsis     [ visualize the learned weighted sum from a downstream checkpoint ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
# -------------#
# -------------#
import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str,
                    help='This has to be a ckpt not a directory.', required=True)
parser.add_argument('--name', type=str, default='', required=False)
parser.add_argument('--out_dir', type=str, default='', required=False)
args = parser.parse_args()

assert os.path.isfile(
    args.ckpt), 'This has to be a ckpt file and not a directory.'
if len(args.name) == 0:
    args.name = args.ckpt.split('/')[-1]  # use the ckpt name
if len(args.out_dir) == 0:
    args.out_dir = '/'.join(args.ckpt.split('/')[:-1])  # use the ckpt dir
else:
    os.mkdir(args.out_dir, exist_ok=True)

ckpt = torch.load(args.ckpt, map_location='cpu')
weights1 = ckpt.get('Featurizer1').get('weights')
weights2 = ckpt.get('Featurizer2').get('weights')
norm_weights1 = F.softmax(weights1, dim=-1).cpu().double().tolist()
norm_weights2 = F.softmax(weights2, dim=-1).cpu().double().tolist()
print('Normalized weights of wavlm+: ', norm_weights1)
print('Normalized weights of hubert: ', norm_weights2)

# plot weights
x = range(1, len(norm_weights1)+1)
plt.bar(x, norm_weights1, 0.3, align='edge', color='deepskyblue')
plt.bar(x, norm_weights2, -0.3, align='edge', color='orange')
# set xticks and ylim
plt.xticks(x, [str(i-1) for i in x])
plt.ylim(0, 1)
# set names
plt.title(f'Distribution of normalized weight - {args.name}')
plt.xlabel('Layer ID (First -> Last)')
plt.ylabel('Weight')
# set legend
colors = {'wavlm': 'deepskyblue', 'hubert': 'orange'}
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label])
           for label in labels]
plt.legend(handles, labels)

plt.savefig(os.path.join(args.out_dir, 'visualize_weight.png'),
            bbox_inches='tight')
