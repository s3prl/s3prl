import torch
import torch.nn as nn




class FrameLevel_Linear(nn.Module):
    def __init__(self, input_dim, class_num, **kwargs):
        super().__init__()
        self.transform = nn.Linear(input_dim, class_num)


    def forward(self, hidden_state, features_len=None):
        logit = self.transform(hidden_state)

        return logit


class UtteranceLevel_Linear(nn.Module):
    def __init__(self, input_dim, class_num, **kwargs):
        super().__init__()
        self.transform = nn.Linear(input_dim, class_num)
    

    def forward(self, hidden_state, features_len=None):
        logit = self.transform(hidden_state)

        return logit


class FrameLevel_1Hidden(nn.Module):
    def __init__(self, input_dim, class_num, hidden_dim, act_fn, **kwargs):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.transform = nn.Linear(hidden_dim, class_num)
        self.act_fn = eval(act_fn)()


    def forward(self, hidden_state, features_len=None):
        hidden_state = self.act_fn(hidden_state)
        hidden_state = self.hidden_layer(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        logit = self.transform(hidden_state)

        return logit



# Pooling Methods

class MeanPooling(nn.Module):

    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()
        # simply MeanPooling / no additional parameters

    def forward(self, feature_BxTxH, features_len, **kwargs):

        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            agg_vec=torch.mean(feature_BxTxH[i][:features_len[i]], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list)

class AttentivePooling(nn.Module):
    ''' Attentive Pooling module incoporate attention mask'''

    def __init__(self, input_dim,**kwargs):
        super(AttentivePooling, self).__init__()

        # Setup
        self.sap_layer = AttentivePoolingModule(input_dim)
    

    def forward(self, feature_BxTxH, features_len):

        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        #Encode
        device = feature_BxTxH.device
        len_masks = torch.lt(torch.arange(features_len.max()).unsqueeze(0).to(device), features_len.unsqueeze(1))
        sap_vec, _ = self.sap_layer(feature_BxTxH, len_masks)


        return sap_vec

class AttentivePoolingModule(nn.Module):
    """
    Implementation of Attentive Pooling 
    """
    def __init__(self, input_dim, **kwargs):
        super(AttentivePoolingModule, self).__init__()
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