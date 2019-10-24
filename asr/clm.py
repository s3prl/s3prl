# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ asr/clm.py ]
#   Synopsis     [ clm for asr]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_Dataloader


# CLM wrapper
class CLM_wrapper(torch.nn.Module):
    def __init__(self,vocab_size,config):
        super(CLM_wrapper, self).__init__()
        # Setup
        self.vocab_size = vocab_size
        self.onehot = torch.nn.Embedding.from_pretrained(torch.eye(vocab_size))
        self.clm = CLM(vocab_size,**config['network'])
        self.loss_weight = config['weight']
        self.optim = getattr(torch.optim,config['optimizer']['type'])
        if config['optimizer']['type']=='Adam':
            self.optim = self.optim(self.clm.parameters(),lr=config['optimizer']['learning_rate'],betas=(0.5, 0.999))
        else:
            self.optim = self.optim(self.clm.parameters(), lr=config['optimizer']['learning_rate'])
        self.gp_lambda = config['optimizer']['gp_lambda']
        self.update_freq = config['optimizer']['update_freq']
        
    def load_text(self,data_config):
        # Independent training set for CLM
        self.train_set = get_Dataloader('text',text_only=True,**data_config)
        self.data_iter = iter(self.train_set)

    def train(self,fake_seq,clm_min_seq_len):
        real_seq = None
        
        # Load real text
        while (real_seq is None) or (real_seq.shape[1]<clm_min_seq_len):
            try:
                real_seq = next(self.data_iter).squeeze(0)
            except StopIteration:
                self.data_iter = iter(self.train_set)
                real_seq = next(self.data_iter).squeeze(0)
        real_seq = self.onehot(real_seq.to(fake_seq.device,dtype=torch.long))

        # For each training step, 
        self.optim.zero_grad()

        score_real = self.loss_weight*self.clm(real_seq).mean()
        score_fake = self.loss_weight*self.clm(fake_seq).mean()
        grad_penal = self.compute_gp(real_seq,fake_seq)
        clm_loss = score_fake - score_real + self.gp_lambda*grad_penal
        
        clm_loss.backward()
        self.optim.step()

        return {'real':score_real,'fake':score_fake},{'gp':grad_penal}
        
    
    def compute_loss(self,seq):
        # Compute adversarial loss for ASR
        return self.loss_weight*self.clm(seq).mean()

    # Calculate gradient penalty, note : <eos> not preserved currently
    # reference : https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
    def compute_gp(self,real_seq,fake_seq):
        bs = int(min(real_seq.size(0),fake_seq.size(0)))
        seq_len = int(min(real_seq.size(1),fake_seq.size(1)))
        dim = int(real_seq.size(2))
        real_seq = real_seq[:bs,:seq_len,:]
        fake_seq = fake_seq[:bs,:seq_len,:]
        
        # Interpolation
        alpha = torch.rand(bs, 1).unsqueeze(2).expand((-1,seq_len,dim)).to(real_seq.device)
        inters = alpha * real_seq + ((1 - alpha) * fake_seq)
        inter = inters.detach()
        inter.requires_grad = True
            
        score_inter = self.clm(inter)
        gradients = torch.autograd.grad(outputs=score_inter, inputs=inter,
                        grad_outputs=torch.ones(score_inter.size()).to(real_seq.device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


# Criticizing Language Model
# please refer to https://arxiv.org/pdf/1811.00787.pdf
class CLM(torch.nn.Module):
    def __init__(self, vocab_size, dim, kernel_size, stride, dropout):
        super(CLM, self).__init__()
        
        # Setting
        dim = list(map(int,dim.split('_')))
        dim = [dim[0]]+dim
        kernel_size = list(map(int,kernel_size.split('_')))
        stride = list(map(int,stride.split('_')))
        dropout = list(map(float,dropout.split('_')))
        self.depth = len(kernel_size)
        
        # Apply CNN for RNN due to 2-order derivative issues
        self.proj = torch.nn.Linear(vocab_size,dim[0])
        for l in range(len(kernel_size)):
            setattr(self,'layer'+str(l),torch.nn.Conv1d(dim[l], dim[l+1], kernel_size[l], stride=stride[l]))
            setattr(self,'drop'+str(l),torch.nn.Dropout(dropout[l]))
        
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.drop_final = torch.nn.Dropout()
        self.score = torch.nn.Linear(dim[-1], 1)

    def forward(self,  x):
        x = F.relu(self.proj(x))
        # BS x T x D -> BS x D x T
        x = x.transpose(1,2)
        for l in range(self.depth):
            x = getattr(self,'layer'+str(l))(x)
            x = getattr(self,'drop'+str(l))(x)
            #x = F.relu(x)
        score = self.score(self.drop_final(self.pooling(x).squeeze(2)))
        return score

