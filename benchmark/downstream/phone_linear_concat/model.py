# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the linear concat model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn


#########
# MODEL #
#########
class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, concat_n_frames, **kwargs):
        super(Model, self).__init__()
        
        assert concat_n_frames > 1, '`concat_n_frames` should be greater than 1.'
        assert concat_n_frames % 2 == 1, '`concat_n_frames must be an odd number.'
        
        # init attributes
        self.concat_n_frames = concat_n_frames
        self.linear = nn.Linear(input_dim*concat_n_frames, output_class_num)          
            

    def _roll(self, x, n, padding='same'):
        # positive n: roll around to the right on the first axis. For example n = 2: [1, 2, 3, 4, 5] -> [4, 5, 1, 2, 3]
        # negative n: roll around to the left on the first axis. For example n = -2: [1, 2, 3, 4, 5] -> [3, 4, 5, 1, 2]
        assert(n != 0)

        if n > 0: # when n is positive (n=2),
            if padding == 'zero':  # set left to zero: [1, 2, 3, 4, 5] -> [0, 0, 1, 2, 3]
                left = torch.zeros_like(x[-n:])
            elif padding == 'same': # set left to same as last: [1, 2, 3, 4, 5] -> [1, 1, 1, 2, 3]
                left = x[0].repeat(n, 1)
            else: # roll over: [1, 2, 3, 4, 5] -> [4, 5, 1, 2, 3]
                left = x[-n:]
            right = x[:-n]

        elif n < 0: # when n is negative (n=-2), 
            if padding == 'zero': # set right to zero: [1, 2, 3, 4, 5] -> [3, 4, 5, 0, 0]
                right = torch.zeros_like(x[:-n])
            elif padding == 'same': # set right to same as last: [1, 2, 3, 4, 5] -> [3, 4, 5, 5, 5]
                right = x[-1].repeat(-n, 1)
            else: # roll over: [1, 2, 3, 4, 5] -> [3, 4, 5, 1, 2]
                right = x[:-n]
            left = x[-n:]
        else:
            raise ValueError('Argument \'n\' should not be set to 0, acceptable range: [-seq_len, 0) and (0, seq_len].')

        return torch.cat((left, right), dim=0)


    def concat_frames(self, features):
        batch_size, seq_len, feature_dim = features.size(0), features.size(1), features.size(2)

        features = features.repeat(1, 1, self.concat_n_frames) # (batch_size, seq_len, feature_dim * concat)
        features = features.view(batch_size, seq_len, self.concat_n_frames, feature_dim).permute(0, 2, 1, 3) # (batch_size, concat, seq_len, feature_dim)
        for b_idx in range(batch_size):
            mid = (self.concat_n_frames // 2)
            for r_idx in range(1, mid+1):
                features[b_idx, mid + r_idx, :] = self._roll(features[b_idx][mid + r_idx], n=r_idx)
                features[b_idx, mid - r_idx, :] = self._roll(features[b_idx][mid - r_idx], n=-r_idx)
        features = features.permute(0, 2, 1, 3).view(batch_size, seq_len, feature_dim * self.concat_n_frames) # (batch_size, seq_len, feature_dim * concat)
        return features


    def forward(self, features):
        concat_features = self.concat_frames(features)
        predicted = self.linear(concat_features)
        return predicted
