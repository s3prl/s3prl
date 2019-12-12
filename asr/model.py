# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ asr/model.py ]
#   Synopsis     [ LAS model for asr]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch 
import random
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np
import math

from utility.asr import Hypothesis
from asr.ctc import CTCPrefixScore

CTC_BEAM_RATIO = 1.5 # DO NOT CHANGE THIS, MAY CAUSE OOM 


class Seq2Seq(nn.Module):
    ''' Seq2Seq model, including Encoder/Decoder(s)'''
    def __init__(self, example_input, output_dim, model_para):
        super(Seq2Seq, self).__init__()
        # Construct Seq2Seq model
        enc_out_dim = int(model_para['encoder']['dim'].split('_')[-1])\
                      *max(1,2*('Bi' in model_para['encoder']['enc_type']))\
                      *max(1,int(model_para['encoder']['sample_rate'].split('_')[-1])\
                           *('concat'== model_para['encoder']['sample_style']))
        
        self.joint_ctc = model_para['optimizer']['joint_ctc']>0
        self.joint_att = model_para['optimizer']['joint_ctc']<1

        # Encoder
        self.encoder = Listener(example_input,**model_para['encoder'])

        # Attention based Decoder
        if self.joint_att:
            self.dec_dim = model_para['decoder']['dim']
            self.attention = Attention(enc_out_dim,self.dec_dim,**model_para['attention'])
            self.decoder = Speller(enc_out_dim+self.dec_dim, **model_para['decoder'])
            self.embed = nn.Embedding(output_dim, self.dec_dim)
            self.char_dim = output_dim
            self.char_trans = nn.Linear(self.dec_dim,self.char_dim)

        # CTC based Decoder
        if self.joint_ctc:
            self.ctc_weight =  model_para['optimizer']['joint_ctc']
            self.ctc_layer = nn.Linear(enc_out_dim,output_dim)

        self.init_parameters()

    def load_lm(self,decode_lm_weight,decode_lm_path,**kwargs):
        # Load RNNLM (for inference only)
        self.rnn_lm = torch.load(decode_lm_path)
        self.rnn_lm.eval()
    
    def clear_att(self):
        self.attention.reset_enc_mem()

    def forward(self, audio_feature, decode_step,tf_rate=0.0,teacher=None,state_len=None):
        bs = audio_feature.shape[0]
        # Encode
        encode_feature,encode_len = self.encoder(audio_feature,state_len)

        ctc_output = None
        att_output = None
        att_maps = None

        # CTC based decoding
        if self.joint_ctc:
            ctc_output = self.ctc_layer(encode_feature)

        # Attention based decoding
        if self.joint_att:
            if teacher is not None:
                teacher = self.embed(teacher)
            
            # Init (init char = <SOS>, reset all rnn state and cell)
            self.decoder.init_rnn(encode_feature)
            self.attention.reset_enc_mem()
            last_char = self.embed(torch.zeros((bs),dtype=torch.long).to(next(self.decoder.parameters()).device))
            output_char_seq = []
            output_att_seq = [[]] * self.attention.num_head
        
            # Decode
            for t in range(decode_step):
                # Attend (inputs current state of first layer, encoded features)
                attention_score,context = self.attention(self.decoder.state_list[0],encode_feature,encode_len)
                # Spell (inputs context + embedded last character)                
                decoder_input = torch.cat([last_char,context],dim=-1)
                dec_out = self.decoder(decoder_input)
                
                # To char
                cur_char = self.char_trans(dec_out)

                # Teacher forcing
                if (teacher is not None):
                    if random.random() <= tf_rate:
                        last_char = teacher[:,t+1,:]
                    else:
                        sampled_char = Categorical(F.softmax(cur_char,dim=-1)).sample()
                        last_char = self.embed(sampled_char)
                else:
                    last_char = self.embed(torch.argmax(cur_char,dim=-1))


                output_char_seq.append(cur_char)
                for head,a in enumerate(attention_score):
                    output_att_seq[head].append(a.cpu())

            att_output = torch.stack(output_char_seq,dim=1)
            att_maps = [torch.stack(att,dim=1) for att in output_att_seq]

        return ctc_output, encode_len, att_output, att_maps

    def init_parameters(self):
        # Reference : https://github.com/espnet/espnet/blob/master/espnet/nets/e2e_asr_th.py
        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 3:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        if self.joint_att:
            self.embed.weight.data.normal_(0, 1)
            for i in range(self.decoder.layer):
                set_forget_bias_to_one(getattr(self.decoder,'layer'+str(i)).bias_ih)
    
    def beam_decode(self, audio_feature, decode_step, state_len,decode_beam_size):
        '''beam decode returns top N hyps for each input sequence'''
        assert audio_feature.shape[0] == 1
        assert self.training == False
        self.decode_beam_size = decode_beam_size
        ctc_beam_size = int(CTC_BEAM_RATIO * self.decode_beam_size)
        
        # Encode
        encode_feature,encode_len = self.encoder(audio_feature,state_len)
        if decode_step==0:
            decode_step = int(encode_len[0])
        
        # Init.
        cur_device = next(self.decoder.parameters()).device
        ctc_output = None
        ctc_state = None
        ctc_prob = 0.0
        ctc_candidates = []
        candidates = None
        att_output = None
        att_maps = None
        lm_hidden = None


        # CTC based decoding
        if self.joint_ctc:
            ctc_output = F.log_softmax(self.ctc_layer(encode_feature),dim=-1)
            ctc_prefix = CTCPrefixScore(ctc_output)
            ctc_state = ctc_prefix.init_state()

        # Attention based decoding
        if self.joint_att:
            # Store attention map if location-aware
            store_att = self.attention.mode == 'loc'
            
            # Init (init char = <SOS>, reset all rnn state and cell)
            self.decoder.init_rnn(encode_feature)
            self.attention.reset_enc_mem()
            last_char = self.embed(torch.zeros((1),dtype=torch.long).to(cur_device))
            last_char_idx = torch.LongTensor([[0]])
            
            # beam search init
            final_outputs, prev_top_outputs, next_top_outputs = [], [], []
            prev_top_outputs.append(Hypothesis(self.decoder.hidden_state, self.embed, output_seq=[], output_scores=[], 
                                               lm_state=None, ctc_prob=0, ctc_state=ctc_state,
                                               att_map = None)) # WIERD BUG here if all args. are not passed...
            # Decode
            for t in range(decode_step):
                for prev_output in prev_top_outputs:
                    
                    # Attention
                    self.decoder.hidden_state = prev_output.decoder_state
                    self.attention.prev_att = None if prev_output.att_map is None else prev_output.att_map.to(cur_device)
                    attention_score,context = self.attention(self.decoder.state_list[0],encode_feature,encode_len)
                    decoder_input = torch.cat([prev_output.last_char,context],dim=-1)
                    dec_out = self.decoder(decoder_input)
                    cur_char = F.log_softmax(self.char_trans(dec_out), dim=-1)

                    # Perform CTC prefix scoring on limited candidates (else OOM easily)
                    if self.joint_ctc:
                        # TODO : Check the performance drop for computing part of candidates only
                        _, ctc_candidates = cur_char.topk(ctc_beam_size)
                        candidates = list(ctc_candidates[0].cpu().numpy())
                        
                        #ctc_prob, ctc_state = ctc_prefix.full_compute(prev_output.outIndex,prev_output.ctc_state,candidates)
                        ctc_prob, ctc_state = ctc_prefix.cheap_compute(prev_output.outIndex,prev_output.ctc_state,candidates)

                        # TODO : study why ctc_char (slightly) > 0 sometimes
                        ctc_char = torch.FloatTensor(ctc_prob - prev_output.ctc_prob).to(cur_device)
                        
                        # Combine CTC score and Attention score (focus on candidates)
                        hack_ctc_char = torch.zeros_like(cur_char).data.fill_(-1000000.0)
                        for idx,char in enumerate(candidates):
                            hack_ctc_char[0,char] = ctc_char[idx]
                        cur_char = (1-self.ctc_weight)*cur_char + self.ctc_weight*hack_ctc_char#ctc_char#
                        cur_char[0,0] = -10000000.0 # Hack to ignore <sos>

                    # Joint RNN-LM decoding
                    if self.decode_lm_weight>0:
                        last_char_idx = prev_output.last_char_idx.to(cur_device)
                        lm_hidden, lm_output = self.rnn_lm(last_char_idx, [1], prev_output.lm_state)
                        cur_char += self.decode_lm_weight * F.log_softmax(lm_output.squeeze(1), dim=-1)

                    # Beam search
                    topv, topi = cur_char.topk(self.decode_beam_size)
                    prev_att_map =  self.attention.prev_att.clone().detach().cpu() if store_att else None 
                    final, top = prev_output.addTopk(topi, topv, self.decoder.hidden_state, att_map=prev_att_map,
                                                     lm_state=lm_hidden,ctc_state=ctc_state,ctc_prob=ctc_prob,
                                                     ctc_candidates=candidates)
                    # Move complete hyps. out
                    if final is not None:
                        final_outputs.append(final)
                        if self.decode_beam_size ==1:
                            return final_outputs
                    next_top_outputs.extend(top)
                
                # Sort for top N beams
                next_top_outputs.sort(key=lambda o: o.avgScore(), reverse=True)
                prev_top_outputs = next_top_outputs[:self.decode_beam_size]
                next_top_outputs = []
            
            final_outputs += prev_top_outputs
            final_outputs.sort(key=lambda o: o.avgScore(), reverse=True)
        
        return final_outputs[:self.decode_beam_size]



# Listener (Encoder)
#     Encodes acoustic feature to latent representation.
# Parameters
#     See config file for more details.
class Listener(nn.Module):
    def __init__(self, example_input, enc_type, sample_rate, sample_style, dim, dropout, rnn_cell):
        super(Listener, self).__init__()
        # Setting
        input_dim = example_input.shape[-1]
        self.enc_type = enc_type
        self.vgg = False
        self.dims = [int(v) for v in dim.split('_')]
        self.sample_rate = [int(v) for v in sample_rate.split('_')]
        self.dropout = [float(v) for v in dropout.split('_')]
        self.sample_style = sample_style

        # Parameters checking
        assert len(self.sample_rate)==len(self.dropout), 'Number of layer mismatch'
        assert len(self.dropout)==len(self.dims), 'Number of layer mismatch'
        self.num_layers = len(self.sample_rate)
        assert self.num_layers>=1,'Listener should have at least 1 layer'

        # Construct Listener
        if 'VGG' in enc_type:
            self.vgg = True
            self.vgg_extractor = VGGExtractor(example_input)
            input_dim = self.vgg_extractor.out_dim

        for l in range(self.num_layers):
            out_dim = self.dims[l]
            sr = self.sample_rate[l]
            drop = self.dropout[l]

            
            if "BiRNN" in enc_type:
                setattr(self, 'layer'+str(l), RNNLayer(input_dim,out_dim, sr, rnn_cell=rnn_cell, dropout_rate=drop,
                                                       bidir=True,sample_style=sample_style))
            elif "RNN" in enc_type:
                setattr(self, 'layer'+str(l), RNNLayer(input_dim,out_dim, sr, rnn_cell=rnn_cell, dropout_rate=drop,
                                                       bidir=False,sample_style=sample_style))
            else:
                raise ValueError('Unsupported Encoder Type: '+enc_type)

            # RNN ouput dim = default output dim x direction x sample rate
            rnn_out_dim = out_dim*max(1,2*('Bi' in enc_type))*max(1,sr*('concat'== sample_style)) 
            setattr(self, 'proj'+str(l),nn.Linear(rnn_out_dim,rnn_out_dim))
            input_dim = rnn_out_dim

    
    def forward(self,input_x,enc_len):
        if self.vgg:
            input_x,enc_len = self.vgg_extractor(input_x,enc_len)
        for l in range(self.num_layers):
            input_x, _,enc_len = getattr(self,'layer'+str(l))(input_x,state_len=enc_len, pack_input=True)
            input_x = torch.tanh(getattr(self,'proj'+str(l))(input_x))
        return input_x,enc_len

# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, input_dim, dim, layer, rnn_cell, dropout):
        super(Speller, self).__init__()
        assert "Cell" in rnn_cell,'Please use Recurrent Cell instead of layer in decoder'
        # Manually forward through Cells if using RNNCell family
        self.layer = layer
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer0 = getattr(nn,rnn_cell)(input_dim,dim)
        for i in range(1,layer):
            setattr(self,'layer'+str(i), getattr(nn,rnn_cell)(dim,dim))
        
        self.state_list = []
        self.cell_list = []
        
        #self.layer = RNNLayer(input_dim,dim, 1, rnn_cell=rnn_cell, layers=layer,
        #                                           dropout_rate=dropout, bidir=False)
    def init_rnn(self,context):
        self.state_list = [torch.zeros(context.shape[0],self.dim).to(context.device)]*self.layer
        self.cell_list = [torch.zeros(context.shape[0],self.dim).to(context.device)]*self.layer

    @property
    def hidden_state(self):
        return [s.clone().detach().cpu() for s in self.state_list], [c.clone().detach().cpu() for c in self.cell_list]

    @hidden_state.setter
    def hidden_state(self, state): # state is a tuple of two list
        device = self.state_list[0].device
        self.state_list = [s.to(device) for s in state[0]]
        self.cell_list = [c.to(device) for c in state[1]]
    
    def forward(self, input_context):
        self.state_list[0],self.cell_list[0] = self.layer0(self.dropout(input_context),(self.state_list[0],self.cell_list[0]))
        for l in range(1,self.layer):
            self.state_list[l],self.cell_list[l] = getattr(self,'layer'+str(l))(self.state_list[l-1],(self.dropout(self.state_list[l]),self.cell_list[l]))
        
        return self.state_list[-1]



# Attention mechanism
# Currently only 'dot' is implemented
# please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
# Input : Decoder state                      with shape [batch size, decoder hidden dimension]
#         Compressed feature from Listner    with shape [batch size, T, listener feature dimension]
# Output: Attention score                    with shape [batch size, T (attention score of each time step)]
#         Context vector                     with shape [batch size, listener feature dimension]
#         (i.e. weighted (by attention score) sum of all timesteps T's feature)

class Attention(nn.Module):  
    def __init__(self, in_dim, dec_dim, att_mode, dim, proj, num_head):
        super(Attention,self).__init__()

        self.mode = att_mode.lower()
        
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)

        # Linear proj. before attention
        self.proj = proj
        if proj:
            self.proj_dim  = dim
            self.phi = nn.Linear(dec_dim,dim*num_head,bias=False)
            self.psi = nn.Linear(in_dim,dim)

        # Layer for merging MHA
        if self.num_head > 1:
            self.merge_head = nn.Linear(in_dim*num_head,in_dim)

        # Location-aware Attetion
        if self.mode == 'loc':
            assert self.proj,"Location-awared attetion requires proj==True"
            assert self.num_head==1
            # TODO : Move this to config
            C = 10
            K = 100
            self.prev_att  = None
            self.loc_conv = nn.Conv1d(1, C, kernel_size=2*K+1, padding=K, bias=False)
            self.loc_proj = nn.Linear(C,dim,bias=False)
            self.gen_energy = nn.Linear(dim, 1)
        
        self.comp_listener_feature = None
    
    def reset_enc_mem(self):
        self.comp_listener_feature = None
        self.state_mask = None
        if self.mode == 'loc':
            self.prev_att = None

    def forward(self, decoder_state, listener_feature, state_len, scale=2.0):
        # Store enc state to save time
        if self.comp_listener_feature is None:
            # Maskout attention score for padded states
            # NOTE: mask MUST have all input > 0 
            self.state_mask = np.zeros((listener_feature.shape[0],listener_feature.shape[1]))
            for idx,sl in enumerate(state_len):
                self.state_mask[idx,sl:] = 1
            self.state_mask = torch.from_numpy(self.state_mask).type(torch.ByteTensor).to(decoder_state.device)
            self.comp_listener_feature =  torch.tanh(self.psi(listener_feature)) if self.proj else listener_feature

        if self.proj:
            comp_decoder_state =  torch.tanh(self.phi(decoder_state))
        else:
            comp_decoder_state = decoder_state

        if self.mode == 'dot':
            if self.num_head == 1:
                energy = torch.bmm(self.comp_listener_feature,comp_decoder_state.unsqueeze(2)).squeeze(dim=2)
                energy.masked_fill_(self.state_mask,-float("Inf"))
                #torch.bmm(comp_decoder_state.unsqueeze(1),self.comp_listener_feature.transpose(1, 2)).squeeze(dim=1)*self.state_mask
                attention_score = [self.softmax(energy*scale)]
                context = torch.bmm(attention_score[0].unsqueeze(1),listener_feature).squeeze(1)
                #torch.sum(listener_feature*attention_score[0].unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1)
            else:
                attention_score =  [ self.softmax(torch.bmm(self.comp_listener_feature,att_querry.unsqueeze(2)).squeeze(dim=2))\
                                    for att_querry in torch.split(comp_decoder_state, self.preprocess_mlp_dim, dim=-1)]
                for idx in range(self.num_head):
                    attention_score[idx].masked_fill_(self.state_mask,-float("inf"))
                    attention_score[idx] = self.softmax(attention_score[idx])
                projected_src = [torch.bmm(att_s.unsqueeze(1),listener_feature).squeeze(1) \
                                for att_s in attention_score]
                context = self.merge_head(torch.cat(projected_src,dim=-1))
        elif self.mode == 'loc':
            if self.prev_att is None:
                # Uniformly init attention
                bs,ts,_ = self.comp_listener_feature.shape
                self.prev_att = torch.zeros((bs,1,ts)).to(self.comp_listener_feature.device)
                for idx,sl in enumerate(state_len):
                    self.prev_att[idx,:,:sl] = 1.0/sl

            comp_decoder_state = comp_decoder_state.unsqueeze(1)
            comp_location_info = torch.tanh(self.loc_proj(self.loc_conv(self.prev_att).transpose(1,2)))
            energy = self.gen_energy(torch.tanh(self.comp_listener_feature+ comp_decoder_state+comp_location_info)).squeeze(2)
            energy.masked_fill_(self.state_mask,-float("inf"))
            attention_score = [self.softmax(energy*scale)]
            self.prev_att = attention_score[0].unsqueeze(1)
            context = torch.bmm(attention_score[0].unsqueeze(1),listener_feature).squeeze(1)
        else:
            # TODO: other attention implementations
            raise ValueError('Unsupported Attention Mode: '+self.mode)
        
        return attention_score,context


# RNN layer
class RNNLayer(nn.Module):
    def __init__(self,in_dim,out_dim, sample_rate, sample_style='drop', layers = 1,
                 rnn_cell='LSTM', dropout_rate=0.0, bidir=True):
        super(RNNLayer, self).__init__()
        self.sample_style = sample_style
        self.sample_rate = sample_rate
        
        self.layer = getattr(nn,rnn_cell.upper())(in_dim,out_dim, bidirectional=bidir, num_layers=layers,
                               dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x,state=None,state_len=None, pack_input=False):
        # Forward RNN
        if pack_input:
            assert state_len is not None, "Please specify seq len for pack_padded_sequence."
            input_x = pack_padded_sequence(input_x, state_len, batch_first=True)
        output,hidden = self.layer(input_x,state)
        if pack_input:
            output,state_len = pad_packed_sequence(output,batch_first=True)
            state_len = state_len.tolist()

        # Perform Downsampling
        if self.sample_rate > 1:
            batch_size,timestep,feature_dim = output.shape

            if self.sample_style =='drop':
                output = output[:,::self.sample_rate,:]
            elif self.sample_style == 'concat':
                if timestep%self.sample_rate != 0: output = output[:,:-(timestep%self.sample_rate),:]
                output = output.contiguous().view(batch_size,int(timestep/self.sample_rate),feature_dim*self.sample_rate)
            else:
                raise ValueError('Unsupported Sample Style: '+self.sample_style)
            if state_len is not None: state_len=[int(s/self.sample_rate) for s in state_len]

        if state_len is not None:
            return output,hidden,state_len
        return output,hidden


# VGG Extractor
# See https://arxiv.org/pdf/1706.02737.pdf for detail
# If VGGextractor is enabled, delta feature should be stack over channel.
class VGGExtractor(nn.Module):
    def __init__(self,example_input):
        super(VGGExtractor, self).__init__()
        in_channel,freq_dim,out_dim = self.check_dim(example_input)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.conv1 = nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(    64, 64, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2) # Half-time dimension
        self.conv3 = nn.Conv2d(    64,128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(   128,128, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2) # Half-time dimension

    def check_dim(self,example_input):
        d = example_input.shape[-1]
        if d%13 == 0:
            # MFCC feature
            return int(d/13),13,(13//4)*128
        elif d%40 == 0:
            # Fbank feature
            return int(d/40),40,(40//4)*128
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,xlen):
        # drop time
        xlen = [x//4 for x in xlen]
        if feature.shape[1]%4 != 0:
            feature = feature[:,:-(feature.shape[1]%4),:].contiguous()
        bs,ts,ds = feature.shape
        # reshape
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,xlen

    def forward(self,feature,xlen):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature,xlen = self.view_input(feature,xlen)
        feature = F.relu(self.conv1(feature))
        feature = F.relu(self.conv2(feature))
        feature = self.pool1(feature) # BSx64xT/2xD/2
        feature = F.relu(self.conv3(feature))
        feature = F.relu(self.conv4(feature))
        feature = self.pool2(feature) # BSx128xT/4xD/4
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1,2)
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0],feature.shape[1],self.out_dim)
        return feature,xlen

