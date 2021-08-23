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

from argparse import Namespace
from s3prl.upstream.mockingjay.model import TransformerEncoder

#########
# MODEL #
#########

class Identity(nn.Module):
    def __init__(self, config):
        super(Identity, self).__init__()
        # simply take mean operator / no additional parameters

    def forward(self, feature, att_mask, head_mask):

        return [feature]

class Mean(nn.Module):

    def __init__(self, out_dim):
        super(Mean, self).__init__()
        # simply take mean operator / no additional parameters

    def forward(self, feature, att_mask):

        ''' 
        Arguments
            feature - [BxTxD]   Acoustic feature with shape 
            att_mask   - [BxTx1]     Attention Mask logits
        '''
        agg_vec_list = []
        for i in range(len(feature)):
            length = torch.nonzero(att_mask[i] < 0, as_tuple=False)[0][0] + 1
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
    def __init__(self, input_dim, agg_module, output_dim, config):
        super(Model, self).__init__()
        
        # agg_module: current support [ "SAP", "Mean" ]
        # init attributes
        self.agg_method = eval(agg_module)(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        
        # two standard transformer encoder layer
        self.model= eval(config['module'])(Namespace(**config['hparams']))
        self.head_mask = [None] * config['hparams']['num_hidden_layers']        


    def forward(self, features, att_mask):

        features = self.model(features,att_mask.unsqueeze(-1), head_mask=self.head_mask, output_all_encoded_layers=False)
        utterance_vector = self.agg_method(features[0], att_mask)
        predicted = self.linear(utterance_vector)
        
        return predicted
        # Use LogSoftmax since self.criterion combines nn.LogSoftmax() and nn.NLLLoss()
