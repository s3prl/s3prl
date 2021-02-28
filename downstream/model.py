import torch
import torch.nn as nn




class FrameLevel_Linear(nn.Module):
    def __init__(self, input_dim, class_num, **kwargs):
        super().__init__()
        self.transform = nn.Linear(input_dim, class_num)


    def forward(self, hidden_state):
        logit = self.transform(hidden_state)

        return logit


class UtteranceLevel_Linear(nn.Module):
    def __init__(self, input_dim, class_num, **kwargs):
        super().__init__()
        self.transform = nn.Linear(input_dim, class_num)
    

    def forward(self, hidden_state):
        logit = self.transform(hidden_state)

        return logit


class FrameLevel_1Hidden(nn.Module):
    def __init__(self, input_dim, class_num, hidden_dim, act_fn, **kwargs):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.transform = nn.Linear(hidden_dim, class_num)
        self.act_fn = eval(act_fn)()


    def forward(self, hidden_state):
        hidden_state = self.act_fn(hidden_state)
        hidden_state = self.hidden_layer(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        logit = self.transform(hidden_state)

        return logit



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

    def __init__(self, input_dim,**kwargs):
        super(AP, self).__init__()

        # Setup
        self.sap_layer = AttentivePooling(input_dim)
    

    def forward(self, feature_BxTxH, att_mask_BxT):

        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            att_mask_BxT  - [BxT]     Attention Mask logits
        '''
        #Encode
        sap_vec, _ = self.sap_layer(feature_BxTxH, att_mask_BxT)


        return sap_vec

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