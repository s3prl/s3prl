# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utils/asr.py ]
#   Synopsis     [ utility pre/post-processing functions for asr]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
#   Reference 2  [ https://groups.google.com/forum/#!msg/librosa/V4Z1HpTKn8Q/1-sMpjxjCSoJ ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import pickle
import torch
import numpy as np
import editdistance as ed
from operator import itemgetter


#################
# ENCODE TARGET #
#################
# Target Encoding Function
# Parameters
#     - input list : list, list of target list
#     - table      : dict, token-index table for encoding (generate one if it's None)
#     - mode       : int, encoding mode ( phoneme / char / subword / word )
#     - max idx    : int, max encoding index (0=<sos>, 1=<eos>, 2=<unk>)
# Return
#     - output list: list, list of encoded targets
#     - output dic : dict, token-index table used during encoding
def encode_target(input_list,table=None,mode='subword',max_idx=500):
    if table is None:
        ### Step 1. Calculate wrd frequency
        table = {}
        for target in input_list:
            for t in target:
                if t not in table:
                    table[t] = 1
                else:
                    table[t] += 1
        ### Step 2. Top k list for encode map
        max_idx = min(max_idx-3,len(table))
        all_tokens = [k for k,v in sorted(table.items(), key = itemgetter(1), reverse = True)][:max_idx]
        table = {'<sos>':0,'<eos>':1}
        if mode == "word": table['<unk>']=2
        for tok in all_tokens:
            table[tok] = len(table)
    ### Step 3. Encode
    output_list = []
    for target in input_list:
        tmp = [0]
        for t in target:
            if t in table:
                tmp.append(table[t])
            else:
                if mode == "word":
                    tmp.append(2)
                else:
                    tmp.append(table['<unk>'])
                    # raise ValueError('OOV error: '+t)
        tmp.append(1)
        output_list.append(tmp)
    return output_list,table


################
# ZERO PADDING #
################
# Feature Padding Function 
# Parameters
#     - x          : list, list of np.array
#     - pad_len    : int, length to pad (0 for max_len in x)      
# Return
#     - new_x      : np.array with shape (len(x),pad_len,dim of feature)
def zero_padding(x,pad_len):
    features = x[0].shape[-1]
    if pad_len is 0: pad_len = max([len(v) for v in x])
    new_x = np.zeros((len(x),pad_len,features))
    for idx,ins in enumerate(x):
        new_x[idx,:min(len(ins),pad_len),:] = ins[:min(len(ins),pad_len),:]
    return new_x


##################
# TARGET PADDING #
##################
# Target Padding Function 
# Parameters
#     - y          : list, list of int
#     - max_len    : int, max length of output (0 for max_len in y)     
# Return
#     - new_y      : np.array with shape (len(y),max_len)
def target_padding(y,max_len):
    if max_len is 0: max_len = max([len(v) for v in y])
    new_y = np.zeros((len(y),max_len),dtype=int)
    for idx,label_seq in enumerate(y):
        new_y[idx,:len(label_seq)] = np.array(label_seq)
    return new_y


##########
# MAPPER #
##########
class Mapper():
    '''Mapper for index2token'''
    def __init__(self,file_path):
        # Find mapping
        with open(os.path.join(file_path,'mapping.pkl'),'rb') as fp:
            self.mapping = pickle.load(fp)
        self.r_mapping = {v:k for k,v in self.mapping.items()}
        symbols = ''.join(list(self.mapping.keys()))
        if '▁' in symbols:
            self.unit = 'subword'
        elif '#' in symbols:
            self.unit = 'phone'
        elif len(self.mapping)<50:
            self.unit = 'char'
        else:
            self.unit = 'word'

    def get_dim(self):
        return len(self.mapping)

    def translate(self,seq,return_string=False):
        new_seq = []
        for c in trim_eos(seq):
            new_seq.append(self.r_mapping[c])
            
        if return_string:
            if self.unit == 'subword':
                new_seq = ''.join(new_seq).replace('<sos>','').replace('<eos>','').replace('▁',' ').lstrip()
            elif self.unit == 'word':
                new_seq = ' '.join(new_seq).replace('<sos>','').replace('<eos>','').lstrip()
            elif self.unit == 'phone':
                new_seq = ' '.join(collapse_phn(new_seq)).replace('<sos>','').replace('<eos>','')
            elif self.unit == 'char':
                new_seq = ''.join(new_seq).replace('<sos>','').replace('<eos>','')
        return new_seq


##############
# HYPOTHESIS #
##############
class Hypothesis:
    '''Hypothesis for beam search decoding.
       Stores the history of label sequence & score 
       Stores the previous decoder state, ctc state, ctc score, lm state and attention map (if necessary)'''
    
    def __init__(self, decoder_state, emb, output_seq=[], output_scores=[], 
                 lm_state=None, ctc_state=None,ctc_prob=0.0,att_map=None):
        assert len(output_seq) == len(output_scores)
        # attention decoder
        self.decoder_state = decoder_state
        self.att_map = att_map
        
        # RNN language model
        self.lm_state = lm_state
        
        # Previous outputs
        self.output_seq = output_seq
        self.output_scores = output_scores
        
        # CTC decoding
        self.ctc_state = ctc_state
        self.ctc_prob = ctc_prob
        
        # Embedding layer for last_char
        self.emb = emb
        

    def avgScore(self):
        '''Return the averaged log probability of hypothesis'''
        assert len(self.output_scores) != 0
        return sum(self.output_scores) / len(self.output_scores)

    def addTopk(self, topi, topv, decoder_state, att_map=None,
                lm_state=None, ctc_state=None, ctc_prob=0.0, ctc_candidates=[]):
        '''Expand current hypothesis with a given beam size'''
        new_hypothesis = []
        term_score = None
        ctc_s,ctc_p = None,None
        beam_size = len(topi[0])
        
        for i in range(beam_size):
            # Detect <eos>
            if topi[0][i].item() == 1:
                term_score = topv[0][i].cpu()
                continue
            
            idxes = self.output_seq[:]     # pass by value
            scores = self.output_scores[:] # pass by value
            idxes.append(topi[0][i].cpu())
            scores.append(topv[0][i].cpu()) 
            if ctc_state is not None:
                #idx = topi[0][i].item() #
                idx = ctc_candidates.index(topi[0][i].item()) #
                ctc_s = ctc_state[idx,:,:]
                ctc_p = ctc_prob[idx]
            new_hypothesis.append(Hypothesis(decoder_state, self.emb,
                                      output_seq=idxes, output_scores=scores, lm_state=lm_state,
                                      ctc_state=ctc_s,ctc_prob=ctc_p,att_map=att_map))
        if term_score is not None:
            self.output_seq.append(torch.tensor(1))
            self.output_scores.append(term_score)
            return self, new_hypothesis
        return None, new_hypothesis

    @property
    def outIndex(self):
        return [i.item() for i in self.output_seq]

    @property
    def last_char_idx(self):
        idx = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        return torch.LongTensor([[idx]])
    @property
    def last_char(self):
        idx = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        return self.emb(torch.LongTensor([idx]).to(next(self.emb.parameters()).device))


################
# CAL ACCURACY #
################
def cal_acc(pred,label):
    pred = np.argmax(pred.cpu().detach(),axis=-1)
    label = label.cpu()
    accs = []
    for p,l in zip(pred,label):
        correct = 0.0
        total_char = 0
        for pp,ll in zip(p,l):
            if ll == 0: break
            correct += int(pp==ll)
            total_char += 1
        accs.append(correct/total_char)
    return sum(accs)/len(accs)


#######################
# CAL CHAR ERROR RATE #
#######################
def cal_cer(pred,label,mapper,get_sentence=False, argmax=True):
    if argmax:
        pred = np.argmax(pred.cpu().detach(),axis=-1)
    label = label.cpu()
    pred = [mapper.translate(p,return_string=True) for p in pred ]
    label = [mapper.translate(l,return_string=True) for l in label]

    if get_sentence:
        return pred,label
    eds = [float(ed.eval(p.split(' '),l.split(' ')))/len(l.split(' ')) for p,l in zip(pred,label)]
    
    return sum(eds)/len(eds)


##################
# DRAW ATTENTION #
##################
# Only draw first attention head
def draw_att(att_list,hyp_txt):
    attmaps = []
    for att,hyp in zip(att_list[0],np.argmax(hyp_txt.cpu().detach(),axis=-1)):
        att_len = len(trim_eos(hyp))
        att = att.detach().cpu()
        attmaps.append(torch.stack([att,att,att],dim=0)[:,:att_len,:]) # +1 for att. @ <eos>
    return attmaps


################
# COLLAPSE PHN #
################
def collapse_phn(seq):
    #phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", 
    #"f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", 
    #"hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
    #"ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

    phonemse_reduce_mapping = {"b":"b", "bcl":"h#", "d":"d", "dcl":"h#", "g":"g", "gcl":"h#", "p":"p", "pcl":"h#", "t":"t", "tcl":"h#", "k":"k", "kcl":"h#", "dx":"dx", "q":"q", "jh":"jh", "ch":"ch", "s":"s", "sh":"sh", "z":"z", "zh":"sh", 
    "f":"f", "th":"th", "v":"v", "dh":"dh", "m":"m", "n":"n", "ng":"ng", "em":"m", "en":"n", "eng":"ng", "nx":"n", "l":"l", "r":"r", "w":"w", "y":"y", 
    "hh":"hh", "hv":"hh", "el":"l", "iy":"iy", "ih":"ih", "eh":"eh", "ey":"ey", "ae":"ae", "aa":"aa", "aw":"aw", "ay":"ay", "ah":"ah", "ao":"aa", "oy":"oy",
    "ow":"ow", "uh":"uh", "uw":"uw", "ux":"uw", "er":"er", "ax":"ah", "ix":"ih", "axr":"er", "ax-h":"ah", "pau":"h#", "epi":"h#", "h#": "h#","<sos>":"<sos>","<unk>":"<unk>","<eos>":"<eos>"}

    return [phonemse_reduce_mapping[c] for c in seq]


############
# TRIM EOS #
############
def trim_eos(seqence):
    new_pred = []
    for char in seqence:
        new_pred.append(int(char))
        if char == 1:
            break
    return new_pred

