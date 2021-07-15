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
import os
import argparse
#-------------#
import torch
import torch.nn.functional as F
#-------------#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, help='This has to be a ckpt not a directory.', required=True)
parser.add_argument('--name', type=str, default='', required=False)
parser.add_argument('--out_dir', type=str, default='', required=False)
args = parser.parse_args()

assert os.path.isfile(args.ckpt), 'This has to be a ckpt file and not a directory.'
if len(args.name) == 0:
    args.name = args.ckpt.split('/')[-1] # use the ckpt name
if len(args.out_dir) == 0:
    args.out_dir = '/'.join(args.ckpt.split('/')[:-1]) # use the ckpt dir
else:
    os.mkdir(args.out_dir, exist_ok=True)

ckpt = torch.load(args.ckpt, map_location='cpu')
weights = ckpt.get('Featurizer').get('weights')
norm_weights = F.softmax(weights, dim=-1).cpu().double().tolist()
print('Normalized weights: ', norm_weights)

# plot weights
x = range(1, len(norm_weights)+1)
plt.bar(x, norm_weights, align='center')

# set xticks and ylim
plt.xticks(x, [str(i) for i in x])
plt.ylim(0, 1)

# set names
plt.title(f'Distribution of normalized weight - {args.name}')
plt.xlabel('Layer ID (First -> Last)')
plt.ylabel('Percentage (%)')

plt.savefig(os.path.join(args.out_dir, 'visualize_weight.png'), bbox_inches='tight')