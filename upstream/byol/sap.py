# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol/sap.py ]
#   Synopsis     [ Implementation of the speaker model ]
#   Author       [ Chen, Yi Chen (https://github.com/grtzsohalf) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/grtzsohalf/SpeechNet/blob/master/bin/sv/model.py ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn


class SAP(nn.Module):
    ''' VERIFI model, including AudioEncoder'''

    def __init__(self, out_dim):
        super(SAP, self).__init__()

        # Setup
        self.linear = nn.Linear(out_dim,out_dim)
        self.act_fn = nn.Tanh()
        self.sap_layer = SelfAttentionPooling(out_dim)
    
    def forward(self, feature, att_mask):

        ''' 
        Arguments
            audio_feature - [BxTxD]   Acoustic feature with shape 
            feature_len   - [BxD]     Length of each sample in a batch
        '''
        #Encode
        feature = self.linear(feature)
        feature = self.act_fn(feature)
        
        # if torch.isnan(feature).any() or torch.isinf(feature).any():
        #     print("invalid value in feature line 41", feature)

        sap_vec = self.sap_layer(feature, att_mask)

        return sap_vec, att_mask


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (N, T, 1)
        
        return:
        utter_rep: size (N, H)
        """
        seq_len = batch_rep.shape[1]
        att_logits = self.W(batch_rep).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep