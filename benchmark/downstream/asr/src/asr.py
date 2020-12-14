# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.util import init_weights, init_gate
from src.module import VGGExtractor, VGGExtractor_LN, VGGExtractor2, FreqVGGExtractor, FreqVGGExtractor2, \
                        RNNLayer, ScaleDotAttention, LocationAwareAttention, liGRU, act_fun, liGRU_layer

class ASR(nn.Module):
    ''' ASR model, including Encoder/Decoder(s)'''
    def __init__(self, input_size, vocab_size, batch_size, ctc_weight, encoder, attention, decoder,  emb_drop=0.0, init_adadelta=True):
        super(ASR, self).__init__()

        # Setup
        assert 0<=ctc_weight<=1
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.enable_ctc = ctc_weight > 0
        self.enable_att = ctc_weight != 1
        self.lm = None

        # Modules
        self.encoder = Encoder(input_size, batch_size, **encoder)
        if self.enable_ctc:
            self.ctc_layer = nn.Sequential(
                nn.Linear(self.encoder.out_dim, 512), 
                nn.Linear(512, vocab_size) 
            )
        if self.enable_att:
            self.dec_dim = decoder['dim']
            self.pre_embed = nn.Embedding(vocab_size, self.dec_dim)
            self.embed_drop = nn.Dropout(emb_drop)
            self.decoder = Decoder(batch_size, self.encoder.out_dim+self.dec_dim, vocab_size, **decoder)
            query_dim = self.dec_dim*self.decoder.layer
            self.attention = Attention(self.encoder.out_dim, query_dim, **attention)

        # Init
    
        """special initialization not work"""
        init_adadelta = False  
        if init_adadelta:
            self.apply(init_weights)
            if self.enable_att:
                for l in range(self.decoder.layer):
                    bias = getattr(self.decoder.layers,'bias_ih_l{}'.format(l))
                    bias = init_gate(bias)
        
        # Orthogonal weight initialisation
        '''
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0) 
        '''        
        
    def set_state(self, prev_state, prev_attn):
        ''' Setting up all memory states for beam decoding'''
        self.decoder.set_state(prev_state)
        self.attention.set_mem(prev_attn)

    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(self.encoder.sample_rate))
        if self.encoder.vgg == 1:
            msg.append('           | VGG Extractor w/ time downsampling rate = 4 in encoder enabled.')
        if self.encoder.vgg == 2:
            msg.append('           | Freq VGG Extractor w/ time downsampling rate = 4 and freq split = {} in encoder enabled.'.format(self.encoder.vgg_freq))
        if self.encoder.vgg == 3:
            msg.append('           | VGG Extractor w/ time downsampling rate = 2 in encoder enabled.'.format(self.encoder.vgg_freq))
        if self.encoder.vgg == 4:
            msg.append('           | Freq VGG Extractor w/ time DS rate = 2, freq split = {}, and low-freq filters = {} in encoder enabled.'.format(self.encoder.vgg_freq, self.encoder.vgg_low_filt))
        if self.encoder.vgg == 5:
            msg.append('           | VGG Extractor w/ Layer normalization and downsampling rate = 4.')

        if self.enable_ctc:
            msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(self.ctc_weight))
        if self.enable_att:
            msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(self.attention.mode,1-self.ctc_weight))
        return msg

    def forward(self, audio_feature, feature_len, decode_step, tf_rate=0.0, teacher=None, 
                      emb_decoder=None, get_dec_state=False, get_logit=False):
        '''
        Arguments
            audio_feature - [BxTxD] Acoustic feature with shape  
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps 
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxLxD] Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return decoder state [BxLxD] for other purpose
        '''
        # Init
        bs = audio_feature.shape[0]
        #print(audio_feature.shape)

        ctc_output, att_output, att_seq = None, None, None
        dec_state = [] if get_dec_state else None

        # Encode
        encode_feature,encode_len = self.encoder(audio_feature,feature_len)
        

        # CTC based decoding
        if self.enable_ctc:
            if get_logit:
                ctc_output = self.ctc_layer(encode_feature)
            else:
                ctc_output = F.log_softmax(self.ctc_layer(encode_feature),dim=-1)

        # Attention based decoding
        if self.enable_att:
            # Init (init char = <SOS>, reset all rnn state and cell)
            self.decoder.init_state(bs) 
            '''init decoder hidden state'''
            self.attention.reset_mem()
            last_char = self.pre_embed(torch.zeros((bs),dtype=torch.long, device=encode_feature.device))
            att_seq, output_seq = [], []

            # Preprocess data for teacher forcing
            if teacher is not None:
                teacher = self.embed_drop(self.pre_embed(teacher))
 
            # Decode
            for t in range(decode_step):
                # Attend (inputs current state of first layer, encoded features)
                attn,context = self.attention(self.decoder.get_query(), encode_feature,encode_len)
 
                # Decode (inputs context + embedded last character)                
                decoder_input = torch.cat([last_char,context],dim=-1)
               
                cur_char, d_state = self.decoder(decoder_input)
                # Prepare output as input of next step
                if (teacher is not None):
                    # Training stage
                    if (tf_rate==1) or (torch.rand(1).item()<=tf_rate):
                        # teacher forcing
                        last_char = teacher[:,t,:]
                    else:
                        # self-sampling (replace by argmax may be another choice)
                        with torch.no_grad():
                            if (emb_decoder is not None) and emb_decoder.apply_fuse:
                                _, cur_prob = emb_decoder(d_state,cur_char,return_loss=False)
                            else:
                                cur_prob = cur_char.softmax(dim=-1)
                            sampled_char = Categorical(cur_prob).sample()
                        last_char = self.embed_drop(self.pre_embed(sampled_char))
                else:
                    # Inference stage
                    if (emb_decoder is not None) and emb_decoder.apply_fuse:
                        _,cur_char = emb_decoder(d_state,cur_char,return_loss=False)
                    # argmax for inference
                    last_char = self.pre_embed(torch.argmax(cur_char,dim=-1))

                # save output of each step
                output_seq.append(cur_char)
                att_seq.append(attn)
                if get_dec_state:
                    dec_state.append(d_state)

            att_output = torch.stack(output_seq,dim=1) # BxTxV
            att_seq = torch.stack(att_seq,dim=2)       # BxNxDtxT
            if get_dec_state:
                dec_state = torch.stack(dec_state,dim=1)

        return ctc_output, encode_len, att_output, att_seq, dec_state
    
    def fix_ctc_layer(self):
        for param in self.ctc_layer.parameters():
            param.requires_grad = False

class Decoder(nn.Module):
    ''' Decoder (a.k.a. Speller in LAS) '''
    # ToDo:　More elegant way to implement decoder 
    def __init__(self, batch_size, input_dim, vocab_size, module, dim, layer, dropout):
        super(Decoder, self).__init__()
        self.in_dim = input_dim
        self.layer = layer
        self.dim = dim
        self.dropout = dropout

        # Init 
        #assert module in ['LSTM','GRU'], NotImplementedError

        self.hidden_state = None
        self.enable_cell = module=='LSTM' #or module=='liGRU'
        
        # Modules
        #print(input_dim)
        if module in ['LSTM', 'GRU', 'liGRU']:
            self.layers = getattr(nn,module)(input_dim,dim, num_layers=layer, dropout=dropout, batch_first=True) # batcb FIRST!
            #self.nolstm = False
            
        '''
        else: # liGRU_layer
            #self.layers = liGRU(input_dim,dim, bidirection= False, dropout=[dropout], layer_norm=[False])       # decode is uni-direction
            self.layers = liGRU_layer(input_dim,dim, batch_size, dropout=dropout, bidirectional= False)       # decode is uni-direction
            self.nolstm = True
        # liGRU(input_dim, dim, bidirection, dropout, layer_norm)
        '''

        self.char_trans = nn.Linear(dim,vocab_size)
        self.final_dropout = nn.Dropout(dropout)
        
        # Orthogonal weight initialisation
        '''
        if not self.nolstm :
            for name, p in self.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(p)
                elif 'bias' in name:
                    nn.init.constant_(p, 0) 
                    '''
                    
        
    def init_state(self, bs):
        ''' Set all hidden states to zeros '''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (torch.zeros((self.layer,bs,self.dim),device=device),
                                 torch.zeros((self.layer,bs,self.dim),device=device))
        else:
            self.hidden_state = torch.zeros((self.layer,bs,self.dim),device=device)
        return self.get_state()

    def set_state(self, hidden_state):
        ''' Set all hidden states/cells, for decoding purpose'''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (hidden_state[0].to(device),hidden_state[1].to(device))
        else:
            self.hidden_state = hidden_state.to(device)

    def get_state(self):
        ''' Return all hidden states/cells, for decoding purpose'''
        if self.enable_cell:
            return (self.hidden_state[0].cpu(),self.hidden_state[1].cpu())
        else:
            return self.hidden_state.cpu()

    def get_query(self):
        ''' Return state of all layers as query for attention '''
        if self.enable_cell:
            return self.hidden_state[0].transpose(0,1).reshape(-1,self.dim*self.layer)
        else:
            return self.hidden_state.transpose(0,1).reshape(-1,self.dim*self.layer)
        '''transpose(0, 1) actually does not do anything'''
    def forward(self, x):
        ''' Decode and transform into vocab '''
        
        x, self.hidden_state = self.layers(x.unsqueeze(1), self.hidden_state)
        x = x.squeeze(1)
        
        char = self.char_trans(self.final_dropout(x))
        return char, x
    
    def fix_layers(self):
        for param in self.parameters():
            param.requires_grad = False


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
        super(Attention,self).__init__()

        # Setup
        self.v_dim = v_dim
        self.dim = dim
        self.mode = mode.lower()
        self.num_head = num_head

        # Linear proj. before attention
        self.proj_q = nn.Linear( q_dim, dim*num_head)
        self.proj_k = nn.Linear( v_dim, dim*num_head)
        self.v_proj = v_proj
        if v_proj:
            self.proj_v = nn.Linear( v_dim, v_dim*num_head)

        # Attention
        if self.mode == 'dot':
            self.att_layer = ScaleDotAttention(temperature, self.num_head)
        elif self.mode == 'loc':
            self.att_layer = LocationAwareAttention(loc_kernel_size, loc_kernel_num, dim, num_head, temperature)
        else:
            raise NotImplementedError

        # Layer for merging MHA
        if self.num_head > 1:
            self.merge_head = nn.Linear(v_dim*num_head, v_dim)
        
        # Stored feature
        self.key = None
        self.value = None
        self.mask = None
        
        # Orthogonal weight initialisation
        '''
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0) 
        '''
    
    def reset_mem(self):
        self.key = None
        self.value = None
        self.mask = None
        self.att_layer.reset_mem()

    def set_mem(self,prev_attn):
        self.att_layer.set_mem(prev_attn)

    def forward(self, dec_state, enc_feat, enc_len):

        # Preprecessing
        bs,ts,_ = enc_feat.shape
        query =  torch.tanh(self.proj_q(dec_state))
        query = query.view(bs, self.num_head, self.dim).view(bs*self.num_head, self.dim) # BNxD

        if self.key is None:
            # Maskout attention score for padded states
            self.att_layer.compute_mask(enc_feat,enc_len.to(enc_feat.device))

            # Store enc state to lower computational cost
            self.key =  torch.tanh(self.proj_k(enc_feat))
            self.value = torch.tanh(self.proj_v(enc_feat)) if self.v_proj else enc_feat # BxTxN
                 
            if self.num_head>1:
                self.key = self.key.view(bs,ts,self.num_head,self.dim).permute(0,2,1,3) # BxNxTxD
                self.key = self.key.contiguous().view(bs*self.num_head,ts,self.dim) # BNxTxD
                if self.v_proj:
                    self.value = self.value.view(bs,ts,self.num_head,self.v_dim).permute(0,2,1,3) # BxNxTxD
                    self.value = self.value.contiguous().view(bs*self.num_head,ts,self.v_dim) # BNxTxD
                else:
                    self.value = self.value.repeat(self.num_head,1,1) #

        # Calculate attention 

        context, attn = self.att_layer(query, self.key, self.value)
        if self.num_head>1:
            context = context.view(bs,self.num_head*self.v_dim)    # BNxD  -> BxND
            context = self.merge_head(context) # BxD
        
        return attn,context

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_layer = 512
        self.bn = nn.BatchNorm1d(self.hidden_layer)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.act = nn.ReLU()
    def forward(self, x):
        '''add DNN at the end of encoder'''
        #print(x.shape)
        x = self.fc1(x)
        x_bn = self.bn(x.view(x.shape[0] * x.shape[1], x.shape[2]))
        x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.act(x)
        x = self.fc2(x)
        x_bn = self.bn(x.view(x.shape[0] * x.shape[1], x.shape[2]))
        x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.act(x)
        return x


class Encoder(nn.Module):
    ''' Encoder (a.k.a. Listener in LAS)
        Encodes acoustic feature to latent representation, see config file for more details.'''
    def __init__(self, input_size, batch_size, vgg, vgg_freq, vgg_low_filt, module, bidirection, dim, dropout, layer_norm, proj, sample_rate, sample_style):
        super(Encoder, self).__init__()

        # Hyper-parameters checking
        self.vgg = vgg
        self.vgg_freq = vgg_freq
        self.vgg_low_filt = vgg_low_filt
        self.sample_rate = 1
        assert len(sample_rate)==len(dropout), 'Number of layer mismatch'
        assert len(dropout)==len(dim), 'Number of layer mismatch'
        num_layers = len(dim)
        assert num_layers>=1,'Encoder should have at least 1 layer'

        # Construct model
        module_list = []
        input_dim = input_size #160
        #print('in', input_dim)
    
        if vgg > 0:
            if vgg == 1:
                vgg_extractor = VGGExtractor(input_size)
            elif vgg == 2:
                vgg_extractor = FreqVGGExtractor(input_size, vgg_freq, vgg_low_filt)
            elif vgg == 3:
                vgg_extractor = VGGExtractor2(input_size)
            elif vgg == 4:
                vgg_extractor = FreqVGGExtractor2(input_size, vgg_freq, vgg_low_filt)
            elif vgg == 5:
                vgg_extractor = VGGExtractor_LN(input_size)
            else:
                raise NotImplementedError('vgg = {} is not available'.format(vgg))
            module_list.append(vgg_extractor)
            input_dim = vgg_extractor.out_dim
            self.sample_rate = self.sample_rate * (4 if vgg < 3 else 2)

        if module in ['LSTM','GRU', 'liGRU']:
            for l in range(num_layers):
                module_list.append(RNNLayer(input_dim, module, dim[l], bidirection, dropout[l], layer_norm[l],
                                            sample_rate[l], sample_style, proj[l], batch_size))
                input_dim = module_list[-1].out_dim
                self.sample_rate = self.sample_rate*sample_rate[l]


            


        self.in_dim = input_size
        self.out_dim = input_dim
        self.layers = nn.ModuleList(module_list)

        # Orthogonal weight initialisation
        '''
        if not self.nolstm :
            for name, p in self.named_parameters():
                if 'weight' in name:                
                    nn.init.orthogonal_(p)
                elif 'bias' in name:
                    nn.init.constant_(p, 0) 
        '''
    def forward(self, input_x, enc_len): # enc_len is feature len
        #print(len(self.layers))
        for _, layer in enumerate(self.layers):
            input_x, enc_len = layer(input_x, enc_len) # is time len

        return input_x, enc_len

    def get_layer_output(self, input_x, enc_len, layer_num=1):
        for i, layer in enumerate(self.layers):
            if i >= layer_num:
                break
            input_x, enc_len = layer(input_x, enc_len)
        return input_x, enc_len
    
    def fix_layers(self, layers):
        for l in layers:
            for param in self.layers[l].parameters():
                param.requires_grad = False

