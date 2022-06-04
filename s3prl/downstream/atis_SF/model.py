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
from torch.distributions.categorical import Categorical
import numpy as np
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

# class Model(nn.Module):
#     def __init__(self, input_dim, agg_module, output_dim, config):
#         super(Model, self).__init__()
        
#         # agg_module: current support [ "SAP", "Mean" ]
#         # init attributes
#         self.agg_method = eval(agg_module)(input_dim)
#         self.linear = nn.Linear(input_dim, output_dim)
        
#         # two standard transformer encoder layer
#         self.model= eval(config['module'])(Namespace(**config['hparams']))
#         self.head_mask = [None] * config['hparams']['num_hidden_layers']        


#     def forward(self, features, att_mask):

#         features = self.model(features,att_mask.unsqueeze(-1), head_mask=self.head_mask, output_all_encoded_layers=False)
#         utterance_vector = self.agg_method(features[0], att_mask)
#         predicted = self.linear(utterance_vector)
        
#         return predicted
        # Use LogSoftmax since self.criterion combines nn.LogSoftmax() and nn.NLLLoss()

class AttenDecoderModel(nn.Module):
    def __init__(self, input_dim, vocab_size, enable_ctc=0., emb_drop=0.0):
        super(AttenDecoderModel, self).__init__()


        decoder = {'module': 'LSTM', 'dim': 768, 'layer': 6, 'dropout': 0.3}
        attention = {'mode': 'loc', 'dim': 320, 'num_head': 4, 'v_proj': False, 'temperature': 0.5, 'loc_kernel_size': 200, 'loc_kernel_num':10}
        self.enable_ctc = enable_ctc
        if self.enable_ctc:
            self.ctc_layer = nn.Linear(input_dim, vocab_size)
            
        self.dec_dim = decoder['dim']
        self.pre_embed = nn.Embedding(vocab_size, self.dec_dim)
        self.embed_drop = nn.Dropout(emb_drop)
        self.decoder = Decoder(
            input_dim+self.dec_dim, vocab_size, **decoder)
        query_dim = self.dec_dim*self.decoder.layer
        self.attention = Attention(
            input_dim, query_dim, **attention)

    def forward(self, encode_feature, decode_step, tf_rate=1.0, teacher=None, 
                      emb_decoder=None, get_dec_state=False, get_logit=False):
        
        ctc_output, att_output, att_seq = None, None, None
        dec_state = [] if get_dec_state else None
        
        bs = encode_feature.shape[0]
        encode_len = torch.IntTensor([len(feat) for feat in encode_feature]).to(device=encode_feature[0].device)
        # CTC based decoding
        if self.enable_ctc:
            ctc_output = F.log_softmax(self.ctc_layer(encode_feature), dim=-1)

        # Attention based decoding
        
        # Init (init char = <SOS>, reset all rnn state and cell)
        self.decoder.init_state(bs)
        self.attention.reset_mem()
        last_char = self.pre_embed(torch.zeros(
            (bs), dtype=torch.long, device=encode_feature.device))
        att_seq, output_seq = [], []

        # Preprocess data for teacher forcing
        if teacher is not None:
            teacher = self.embed_drop(self.pre_embed(teacher))

        # Decode
        for t in range(decode_step):
            # Attend (inputs current state of first layer, encoded features)
            attn, context = self.attention(
                self.decoder.get_query(), encode_feature, encode_len)
            # Decode (inputs context + embedded last character)
            decoder_input = torch.cat([last_char, context], dim=-1)
            cur_char, d_state = self.decoder(decoder_input)
            # Prepare output as input of next step
            if (teacher is not None):
                # Training stage
                if (tf_rate == 1) or (torch.rand(1).item() <= tf_rate):
                    # teacher forcing
                    last_char = teacher[:, t, :]
                else:
                    # self-sampling (replace by argmax may be another choice)
                    with torch.no_grad():
                        if (emb_decoder is not None) and emb_decoder.apply_fuse:
                            _, cur_prob = emb_decoder(
                                d_state, cur_char, return_loss=False)
                        else:
                            cur_prob = cur_char.softmax(dim=-1)
                        sampled_char = Categorical(cur_prob).sample()
                    last_char = self.embed_drop(
                        self.pre_embed(sampled_char))
            else:
                # Inference stage
                if (emb_decoder is not None) and emb_decoder.apply_fuse:
                    _, cur_char = emb_decoder(
                        d_state, cur_char, return_loss=False)
                # argmax for inference
                last_char = self.pre_embed(torch.argmax(cur_char, dim=-1))

            # save output of each step
            output_seq.append(cur_char)
            att_seq.append(attn)
            if get_dec_state:
                dec_state.append(d_state)

        att_output = torch.stack(output_seq, dim=1)  # BxTxV
        att_seq = torch.stack(att_seq, dim=2)       # BxNxDtxT
        if get_dec_state:
            dec_state = torch.stack(dec_state, dim=1)

        return ctc_output, encode_len, att_output, att_seq, dec_state

'''LAS'''

class Attention(nn.Module):
    ''' Attention mechanism
        please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
        Input : Decoder state                      with shape [batch size, decoder hidden dimension]
                Compressed feature from Encoder    with shape [batch size, T, encoder feature dimension]
        Output: Attention score                    with shape [batch size, num head, T (attention score of each time step)]
                Context vector                     with shape [batch size, encoder feature dimension]
                (i.e. weighted (by attention score) sum of all timesteps T's feature) '''

    def __init__(self, v_dim, q_dim, mode, dim, num_head, temperature, v_proj,
                 loc_kernel_size, loc_kernel_num):
        super(Attention, self).__init__()

        # Setup
        self.v_dim = v_dim
        self.dim = dim
        self.mode = mode.lower()
        self.num_head = num_head

        # Linear proj. before attention
        self.proj_q = nn.Linear(q_dim, dim*num_head)
        self.proj_k = nn.Linear(v_dim, dim*num_head)
        self.v_proj = v_proj
        if v_proj:
            self.proj_v = nn.Linear(v_dim, v_dim*num_head)

        # Attention
        if self.mode == 'dot':
            self.att_layer = ScaleDotAttention(temperature, self.num_head)
        elif self.mode == 'loc':
            self.att_layer = LocationAwareAttention(
                loc_kernel_size, loc_kernel_num, dim, num_head, temperature)
        else:
            raise NotImplementedError

        # Layer for merging MHA
        if self.num_head > 1:
            self.merge_head = nn.Linear(v_dim*num_head, v_dim)

        # Stored feature
        self.key = None
        self.value = None
        self.mask = None

    def reset_mem(self):
        self.key = None
        self.value = None
        self.mask = None
        self.att_layer.reset_mem()

    def set_mem(self, prev_attn):
        self.att_layer.set_mem(prev_attn)

    def forward(self, dec_state, enc_feat, enc_len):

        # Preprecessing
        bs, ts, _ = enc_feat.shape
        query = torch.tanh(self.proj_q(dec_state))
        query = query.view(bs, self.num_head, self.dim).view(
            bs*self.num_head, self.dim)  # BNxD

        if self.key is None:
            # Maskout attention score for padded states
            self.att_layer.compute_mask(enc_feat, enc_len.to(enc_feat.device))

            # Store enc state to lower computational cost
            self.key = torch.tanh(self.proj_k(enc_feat))
            self.value = torch.tanh(self.proj_v(
                enc_feat)) if self.v_proj else enc_feat  # BxTxN

            if self.num_head > 1:
                self.key = self.key.view(bs, ts, self.num_head, self.dim).permute(
                    0, 2, 1, 3)  # BxNxTxD
                self.key = self.key.contiguous().view(bs*self.num_head, ts, self.dim)  # BNxTxD
                if self.v_proj:
                    self.value = self.value.view(
                        bs, ts, self.num_head, self.v_dim).permute(0, 2, 1, 3)  # BxNxTxD
                    self.value = self.value.contiguous().view(
                        bs*self.num_head, ts, self.v_dim)  # BNxTxD
                else:
                    self.value = self.value.repeat(self.num_head, 1, 1)

        # Calculate attention
        context, attn = self.att_layer(query, self.key, self.value)
        if self.num_head > 1:
            context = context.view(
                bs, self.num_head*self.v_dim)    # BNxD  -> BxND
            context = self.merge_head(context)  # BxD

        return attn, context


class Decoder(nn.Module):
    ''' Decoder (a.k.a. Speller in LAS) '''
    # ToDo:　More elegant way to implement decoder

    def __init__(self, input_dim, vocab_size, module, dim, layer, dropout):
        super(Decoder, self).__init__()
        self.in_dim = input_dim
        self.layer = layer
        self.dim = dim
        self.dropout = dropout

        # Init
        assert module in ['LSTM', 'GRU'], NotImplementedError
        self.hidden_state = None
        self.enable_cell = module == 'LSTM'

        # Modules
        self.layers = getattr(nn, module)(
            input_dim, dim, num_layers=layer, dropout=dropout, batch_first=True)
        self.char_trans = nn.Linear(dim, vocab_size)
        self.final_dropout = nn.Dropout(dropout)

    def init_state(self, bs):
        ''' Set all hidden states to zeros '''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (torch.zeros((self.layer, bs, self.dim), device=device),
                                 torch.zeros((self.layer, bs, self.dim), device=device))
        else:
            self.hidden_state = torch.zeros(
                (self.layer, bs, self.dim), device=device)
        return self.get_state()

    def set_state(self, hidden_state):
        ''' Set all hidden states/cells, for decoding purpose'''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (hidden_state[0].to(
                device), hidden_state[1].to(device))
        else:
            self.hidden_state = hidden_state.to(device)

    def get_state(self):
        ''' Return all hidden states/cells, for decoding purpose'''
        if self.enable_cell:
            return (self.hidden_state[0].cpu(), self.hidden_state[1].cpu())
        else:
            return self.hidden_state.cpu()

    def get_query(self):
        ''' Return state of all layers as query for attention '''
        if self.enable_cell:
            return self.hidden_state[0].transpose(0, 1).reshape(-1, self.dim*self.layer)
        else:
            return self.hidden_state.transpose(0, 1).reshape(-1, self.dim*self.layer)

    def forward(self, x):
        ''' Decode and transform into vocab '''
        if not self.training:
            self.layers.flatten_parameters()
        x, self.hidden_state = self.layers(x.unsqueeze(1), self.hidden_state)
        x = x.squeeze(1)
        char = self.char_trans(self.final_dropout(x))
        return char, x

class BaseAttention(nn.Module):
    ''' Base module for attentions '''

    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None

    def set_mem(self, prev_att):
        pass

    def compute_mask(self, k, k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs, ts, _ = k.shape
        self.mask = np.zeros((bs, self.num_head, ts))
        for idx, sl in enumerate(k_len):
            self.mask[idx, :, sl:] = 1  # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(
            k_len.device, dtype=torch.bool).view(-1, ts)  # BNxT

    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf)
        attn = self.softmax(attn)  # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(
            1)  # BNxT x BNxTxD-> BNxD
        return output, attn


class ScaleDotAttention(BaseAttention):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, num_head):
        super().__init__(temperature, num_head)

    def forward(self, q, k, v):
        ts = k.shape[1]
        energy = torch.bmm(q.unsqueeze(1), k.transpose(
            1, 2)).squeeze(1)  # BNxD * BNxDxT = BNxT
        output, attn = self._attend(energy, v)

        attn = attn.view(-1, self.num_head, ts)  # BNxT -> BxNxT

        return output, attn


class LocationAwareAttention(BaseAttention):
    ''' Location-Awared Attention '''

    def __init__(self, kernel_size, kernel_num, dim, num_head, temperature):
        super().__init__(temperature, num_head)
        self.prev_att = None
        self.loc_conv = nn.Conv1d(
            num_head, kernel_num, kernel_size=2*kernel_size+1, padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(kernel_num, dim, bias=False)
        self.gen_energy = nn.Linear(dim, 1)
        self.dim = dim

    def reset_mem(self):
        super().reset_mem()
        self.prev_att = None

    def set_mem(self, prev_att):
        self.prev_att = prev_att

    def forward(self, q, k, v):
        bs_nh, ts, _ = k.shape
        bs = bs_nh//self.num_head

        # Uniformly init prev_att
        if self.prev_att is None:
            self.prev_att = torch.zeros((bs, self.num_head, ts)).to(k.device)
            for idx, sl in enumerate(self.k_len):
                self.prev_att[idx, :, :sl] = 1.0/sl

        # Calculate location context
        loc_context = torch.tanh(self.loc_proj(self.loc_conv(
            self.prev_att).transpose(1, 2)))  # BxNxT->BxTxD
        loc_context = loc_context.unsqueeze(1).repeat(
            1, self.num_head, 1, 1).view(-1, ts, self.dim)   # BxNxTxD -> BNxTxD
        q = q.unsqueeze(1)  # BNx1xD

        # Compute energy and context
        energy = self.gen_energy(torch.tanh(
            k+q+loc_context)).squeeze(2)  # BNxTxD -> BNxT
        output, attn = self._attend(energy, v)
        attn = attn.view(bs, self.num_head, ts)  # BNxT -> BxNxT
        self.prev_att = attn

        return output, attn


'''transformer'''
from torch.nn import Transformer, TransformerDecoder, TransformerDecoderLayer
from torch import Tensor
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers=3,
                 num_decoder_layers=6,
                 emb_size=512,
                 nhead=4,
                 tgt_vocab_size=600,
                 dim_feedforward=2048,
                 dropout=0.1, 
                 is_unit=False,
                 unit_size=None,
                 is_dual_decoder=False,
                 is_bart_decoder=False,
                 pass_extra_encoder=False):
        super(Seq2SeqTransformer, self).__init__()

        self.is_bart_decoder = is_bart_decoder


        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        self.is_unit = is_unit
        
        # replace decoder to pre-trained bart decoder
        if self.is_bart_decoder: 
            from transformers import AutoModelForSeq2SeqLM

            decoder_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')
            del self.transformer.decoder
            self.transformer.decoder = decoder_model.model.decoder
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
            self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
            

        # for ctc unit predition
        unit_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=emb_size, batch_first=True)
        # add extra encoder layer
        self.unit_encoder = nn.TransformerEncoder(unit_encoder_layer, num_encoder_layers)
        self.ctc_layer = nn.Linear(emb_size, unit_size)

        self.pass_extra_encoder = pass_extra_encoder
        self.is_dual_decoder = is_dual_decoder
        if self.is_dual_decoder: 
            decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=512, batch_first=True)
            self.unit_decoder = TransformerDecoder(decoder_layer, 3)
            self.unit_tok_emb = TokenEmbedding(unit_size, emb_size)
            self.unit_positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
            self.unit_generator = nn.Linear(emb_size, unit_size)    

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor, 
                unit=None, 
                ctc_weight=0,
                unit_mask=None):
        
        if not self.pass_extra_encoder:
            memory = self.transformer.encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)

        ctc_output = None
        if self.is_unit and ctc_weight > 0.0: 
            unit_ctc_output = self.unit_encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)
            ctc_output = F.log_softmax(self.ctc_layer(unit_ctc_output), dim=-1)

        unit_outs = None
        if self.is_dual_decoder: 
            unit = self.unit_positional_encoding(self.unit_tok_emb(unit))
            unit_outs = self.unit_decoder(unit, memory, tgt_mask=unit_mask)
        
        # main task
        if ctc_weight < 1.0: 
            tgt = self.positional_encoding(self.tgt_tok_emb(trg))
            if self.is_bart_decoder: 
                if self.pass_extra_encoder:
                    outs = self.transformer.decoder(input_ids=trg, encoder_hidden_states=src, encoder_attention_mask=src_padding_mask).last_hidden_state
                else:
                    outs = self.transformer.decoder(input_ids=trg, encoder_hidden_states=memory).last_hidden_state
                
            else:
                outs = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
            # dual decoder for unit
            if unit_outs is not None: 
                return self.generator(outs), ctc_output, self.unit_generator(unit_outs)
            else: 
                return self.generator(outs), ctc_output, None
        else: 
            if unit_outs is not None: 
                return None, ctc_output, self.unit_generator(unit_outs)
            else: 
                return None, ctc_output, None 

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, src_mask=None):        
        if self.is_bart_decoder: 
            if self.pass_extra_encoder:
                return self.transformer.decoder(input_ids=tgt.transpose(1, 0), encoder_hidden_states=memory, encoder_attention_mask=src_mask).last_hidden_state
            else:
                return self.transformer.decoder(input_ids=tgt.transpose(1, 0), encoder_hidden_states=memory).last_hidden_state
        else:
            tgt = self.positional_encoding(self.tgt_tok_emb(tgt))
            tgt = tgt.view(tgt.shape[1], tgt.shape[0], tgt.shape[2])
        
            return self.transformer.decoder(tgt, memory, tgt_mask) 
            

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

PAD_IDX = 0
EOS_IDX = 1
def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    tgt_padding_mask = (tgt == PAD_IDX)
    return src_mask, tgt_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol=2, pass_extra_encoder=False):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    if not pass_extra_encoder:
        memory = model.encode(src, src_mask)
        memory = memory.to(DEVICE)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for _ in range(max_len-1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        if pass_extra_encoder:
            out = model.decode(ys, src, tgt_mask, src_mask=src_mask)
        else: 
            out = model.decode(ys, memory, tgt_mask)

        prob = model.generator(out[:, -1])
        
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word[0].item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).fill_(next_word).type(torch.long).to(DEVICE)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys