# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the linear model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
import torch.nn.functional as F

import IPython
import pdb
from argparse import Namespace
from upstream.mockingjay.model import TransformerEncoder

#########
# MODEL #
#########

class Identity(nn.Module):
    def __init__(self, config, **kwargs):
        super(Identity, self).__init__()
        # simply take mean operator / no additional parameters

    def forward(self, feature, att_mask, head_mask, **kwargs):

        return [feature]

class Mean(nn.Module):

    def __init__(self, out_dim):
        super(Mean, self).__init__()
        self.act_fn = nn.ReLU()
        self.linear = nn.Linear(out_dim, out_dim)
        # simply take mean operator / no additional parameters

    def forward(self, feature, att_mask):

        ''' 
        Arguments
            feature - [BxTxD]   Acoustic feature with shape 
            att_mask   - [BxTx1]     Attention Mask logits
        '''
        feature=self.linear(self.act_fn(feature))
        agg_vec_list = []
        for i in range(len(feature)):
            if torch.nonzero(att_mask[i] < 0, as_tuple=False).size(0) == 0:
                length = len(feature[i])
            else:
                length = torch.nonzero(att_mask[i] < 0, as_tuple=False)[0] + 1
            agg_vec=torch.mean(feature[i][:length], dim=0)
            agg_vec_list.append(agg_vec)
        return torch.stack(agg_vec_list)

class SAP(nn.Module):
    ''' Self Attention Pooling module incoporate attention mask'''

    def __init__(self, out_dim):
        super(SAP, self).__init__()

        # Setup
        self.act_fn = nn.Tanh()
        self.sap_layer = SelfAttentionPooling(out_dim)
    
    def forward(self, feature, att_mask):

        ''' 
        Arguments
            feature - [BxTxD]   Acoustic feature with shape 
            att_mask   - [BxTx1]     Attention Mask logits
        '''
        #Encode
        feature = self.act_fn(feature)
        sap_vec = self.sap_layer(feature, att_mask)

        return sap_vec

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
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
        softmax = nn.functional.softmax
        att_logits = self.W(batch_rep).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Model(nn.Module):
    def __init__(self, input_dim, agg_module, config):
        super(Model, self).__init__()
        
        # agg_module: current support [ "SAP", "Mean" ]
        # init attributes
        self.agg_method = eval(agg_module)(input_dim)
        
        # two standard transformer encoder layer
        self.model= eval(config['module'])(config=Namespace(**config['hparams']),)
        self.head_mask = [None] * config['hparams']['num_hidden_layers']         


    def forward(self, features, att_mask):
        features = self.model(features,att_mask[:,None,None], head_mask=self.head_mask, output_all_encoded_layers=False)
        utterance_vector = self.agg_method(features[0], att_mask)
        
        return utterance_vector


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)