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

def decide_utter_input_dim(agg_module_name, input_dim, agg_dim):		
    if agg_module_name =="ASP":		
        utter_input_dim = input_dim*2		
    elif agg_module_name == "SP":		
        # after aggregate to utterance vector, the vector hidden dimension will become 2 * aggregate dimension.		
        utter_input_dim = agg_dim*2		
    elif agg_module_name == "MP":		
        utter_input_dim = agg_dim		
    else:		
        utter_input_dim = input_dim		
    return utter_input_dim

# Pooling Methods

class MP(nn.Module):

    def __init__(self, **kwargs):
        super(MP, self).__init__()
        # simply MeanPooling / no additional parameters

    def forward(self, feature_BxTxH, att_mask_BxT, **kwargs):

        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            att_mask_BxT  - [BxT]     Attention Mask logits
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            if torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False).size(0) == 0:
                length = len(feature_BxTxH[i])
            else:
                length = torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False)[0] + 1
            agg_vec=torch.mean(feature_BxTxH[i][:length], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list)

class AP(nn.Module):
    ''' Attentive Pooling module incoporate attention mask'''

    def __init__(self, out_dim, input_dim):
        super(AP, self).__init__()

        # Setup
        self.linear = nn.Linear(input_dim, out_dim)
        self.sap_layer = AttentivePooling(out_dim)
        self.act_fn=nn.ReLU()
    
    def forward(self, feature_BxTxH, att_mask_BxT):

        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            att_mask_BxT  - [BxT]     Attention Mask logits
        '''
        #Encode
        feature_BxTxH = self.linear(feature_BxTxH)
        sap_vec, _ = self.sap_layer(feature_BxTxH, att_mask_BxT)

        return sap_vec

class ASP(nn.Module):
    ''' Attentive Statistic Pooling module incoporate attention mask'''

    def __init__(self, out_dim, input_dim):
        super(ASP, self).__init__()

        # Setup
        self.linear = nn.Linear(input_dim, out_dim)
        self.ap_layer = AttentivePooling(out_dim)

    
    def forward(self, feature_BxTxH, att_mask_BxT):

        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            att_mask_BxT  - [BxT]     Attention Mask logits
        '''
        #Encode
        feature_BxTxH = self.linear(feature_BxTxH)
        sap_vec, att_w = self.ap_layer(feature_BxTxH, att_mask_BxT)
        variance = torch.sqrt(torch.sum(att_w * feature_BxTxH * feature_BxTxH, dim=1) - sap_vec**2 + 1e-8)
        statistic_pooling = torch.cat([sap_vec, variance], dim=-1)

        return statistic_pooling

class SP(nn.Module):
    ''' Statistic Pooling incoporate attention mask'''

    def __init__(self, out_dim, input_dim, *kwargs):
        super(SP, self).__init__()

        # Setup
        self.mp_layer = MP()
    
    def forward(self, feature_BxTxH, att_mask_BxT):

        ''' 
        Arguments
            feature - [BxTxH]   Acoustic feature with shape 
            att_mask- [BxT]     Attention Mask logits
        '''
        #Encode
        mean_vec = self.mp_layer(feature_BxTxH, att_mask_BxT)
        variance_vec_list = []
        for i in range(len(feature_BxTxH)):
            if torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False).size(0) == 0:
                length = len(feature_BxTxH[i])
            else:
                length = torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False)[0] + 1
            variances = torch.std(feature_BxTxH[i][:length], dim=-2)
            variance_vec_list.append(variances)
        var_vec = torch.stack(variance_vec_list)

        statistic_pooling = torch.cat([mean_vec, var_vec], dim=-1)

        return statistic_pooling

class AttentivePooling(nn.Module):
    """
    Implementation of Attentive Pooling 
    """
    def __init__(self, input_dim, **kwargs):
        super(AttentivePooling, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = nn.ReLU()
        self.softmax = nn.functional.softmax
    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (B, T, 1)
        
        return:
        utter_rep: size (B, H)
        """
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w


# General Interface
class Model(nn.Module):
    def __init__(self, input_dim, agg_dim, agg_module_name, module_name, utterance_module_name, hparams):
        super(Model, self).__init__()
        
        # support for XVector(standard architecture), Identity (do nothing)
        # Framewise FeatureExtractor
        extractor_config = {**hparams, **{"input_dim": input_dim}}
        self.framelevel_feature_extractor= eval(module_name)(**extractor_config)

        # agg_module: 
        # current support:
        # [ "AP" (Attentive Pooling), "MP" (Mean Pooling), "SP" (Statistic Pooling), "SAP" (Statistic Attentive Pooling) ]
        agg_module_config = {"out_dim": input_dim, "input_dim": agg_dim}
        self.agg_method = eval(agg_module_name)(**agg_module_config)

        utterance_input_dim = decide_utter_input_dim(agg_module_name=agg_module_name, agg_dim=agg_dim, input_dim=input_dim)

        # after extract utterance level vector, put it to utterance extractor (XVector Architecture)
        utterance_extractor_config = {"input_dim": utterance_input_dim,"out_dim": input_dim}
        self.utterancelevel_feature_extractor= eval(utterance_module_name)(**utterance_extractor_config)

    def forward(self, features_BxTxH, att_mask_BxT):
        
        features_BxTxH = self.framelevel_feature_extractor(features_BxTxH, att_mask_BxT[:,None,None])
        utterance_vector = self.agg_method(features_BxTxH, att_mask_BxT)
        utterance_vector = self.utterancelevel_feature_extractor(utterance_vector)
        
        return utterance_vector
    
    def inference(self, features_BxTxH, att_mask_BxT):
        
        features_BxTxH = self.framelevel_feature_extractor(features_BxTxH, att_mask_BxT[:,None,None])
        utterance_vector = self.agg_method(features_BxTxH, att_mask_BxT)
        utterance_vector = self.utterancelevel_feature_extractor.inference(utterance_vector)

        return utterance_vector

class UtteranceExtractor(nn.Module):
    def __init__(self, input_dim, out_dim, **kwargs):
        super(UtteranceExtractor,self).__init__()
        self.linear1 = nn.Linear(input_dim,out_dim)
        self.linear2 = nn.Linear(out_dim,out_dim)
        self.act_fn = nn.ReLU()
    def forward(self, x_BxH):
        hid_BxH = self.linear1(x_BxH)
        hid_BxH = self.act_fn(hid_BxH)
        hid_BxH = self.linear2(hid_BxH)
        hid_BxH = self.act_fn(hid_BxH)

        return hid_BxH
    
    def inference(self, feature_BxH):
        hid_BxH = self.linear1(feature_BxH)
        hid_BxH = self.act_fn(hid_BxH)

        return hid_BxH

# General Interface
class UtteranceIdentity(nn.Module):
    def __init__(self, input_dim, out_dim, **kwargs):
        super(UtteranceIdentity,self).__init__()
        self.linear=nn.Linear(input_dim, out_dim)
        self.act_fn = nn.ReLU()
    def forward(self, x_BxH):
        hid_BxH = self.act_fn(x_BxH)
        hid_BxH = self.linear(hid_BxH)
        hid_BxH = self.act_fn(hid_BxH)

        return hid_BxH
    
    def inference(self, x_BxH):
        hid_BxH = self.act_fn(x_BxH)
        hid_BxH = self.linear(hid_BxH)
        hid_BxH = self.act_fn(hid_BxH)

        return hid_BxH
    

class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()
        # simply forward / no additional parameters

    def forward(self, feature_BxTxH, att_mask_BxTx1x1, **kwargs):

        return feature_BxTxH

class XVector(nn.Module):
    def __init__(self, input_dim, agg_dim, dropout_p, batch_norm, **kwargs):
        super(XVector, self).__init__()
        # simply take mean operator / no additional parameters
        self.module = nn.Sequential(
            TDNN(input_dim=input_dim, output_dim=input_dim, context_size=5, dilation=1, batch_norm=batch_norm, dropout_p=dropout_p),
            TDNN(input_dim=input_dim, output_dim=input_dim, context_size=3, dilation=2, batch_norm=batch_norm, dropout_p=dropout_p),
            TDNN(input_dim=input_dim, output_dim=input_dim, context_size=3, dilation=3, batch_norm=batch_norm, dropout_p=dropout_p),
            TDNN(input_dim=input_dim, output_dim=input_dim, context_size=1, dilation=1, batch_norm=batch_norm, dropout_p=dropout_p),
            TDNN(input_dim=input_dim, output_dim=agg_dim, context_size=1, dilation=1, batch_norm=batch_norm, dropout_p=dropout_p),
        )

    def forward(self, feature_BxTxH, att_mask_BxTx1x1, **kwargs):

        feature_BxTxH=self.module(feature_BxTxH)
        return feature_BxTxH


class AMSoftmaxLoss(nn.Module):

    def __init__(self, hidden_dim, speaker_num, s=30.0, m=0.4, **kwargs):
        '''
        AM Softmax Loss
        '''
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.speaker_num = speaker_num
        self.W = torch.nn.Parameter(torch.randn(hidden_dim, speaker_num), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x_BxH, labels_B):
        '''
        x shape: (B, H)
        labels shape: (B)
        '''
        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num
        
        W = F.normalize(self.W, dim=0)

        x_BxH = F.normalize(x_BxH, dim=1)

        wf = torch.mm(x_BxH, W)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels_B]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels_B)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

class SoftmaxLoss(nn.Module):
    
    def __init__(self, hidden_dim, speaker_num, **kwargs):
        '''
        Softmax Loss
        '''
        super(SoftmaxLoss, self).__init__()
        self.fc = nn.Linear(hidden_dim, speaker_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x_BxH, labels_B):
        '''
        x shape: (B, H)
        labels shape: (B)
        '''
        logits_BxSpn = self.fc(x_BxH)
        loss = self.loss(logits_BxSpn, labels_B)
        
        return loss

class AAMSoftmaxLoss(nn.Module):
    def __init__(self, hidden_dim, speaker_num, s=15, m=0.3, easy_margin=False, **kwargs):
        super(AAMSoftmaxLoss, self).__init__()
        import math

        self.test_normalize = True
        
        self.m = m
        self.s = s
        self.speaker_num = speaker_num
        self.hidden_dim = hidden_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(speaker_num, hidden_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x_BxH, labels_B):

        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num
        
        # cos(theta)
        cosine = F.linear(F.normalize(x_BxH), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels_B.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, labels_B)
        return loss

class TDNN(nn.Module):
        
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=False,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x_BxTxH):
        '''
        input: size (batch B, seq_len T, input_features H)
        outpu: size (batch B, new_seq_len T*, output_features H)
        '''

        _, _, d = x_BxTxH.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x_BxTxH = x_BxTxH.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x_BxTxH = F.unfold(
                        x_BxTxH, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x_BxTxH = x_BxTxH.transpose(1,2)
        x_BxTxH = self.kernel(x_BxTxH)
        x_BxTxH = self.nonlinearity(x_BxTxH)
        
        if self.dropout_p:
            x_BxTxH = self.drop(x_BxTxH)

        if self.batch_norm:
            x_BxTxH = x_BxTxH.transpose(1,2)
            x_BxTxH = self.bn(x_BxTxH)
            x_BxTxH = x_BxTxH.transpose(1,2)

        return x_BxTxH
