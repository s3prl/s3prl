import math
import time
import torch
import numpy as np
from torch import nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
'''new class for label smoothing loss'''
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        # true_dist = pred.data.clone()
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1)) # uniform distribution u = 1/ numClass
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))




class Timer():
    ''' Timer for recording training time distribution. '''
    def __init__(self):
        self.prev_t = time.time()
        self.clear()

    def set(self):
        self.prev_t = time.time()

    def cnt(self,mode):
        self.time_table[mode] += time.time()-self.prev_t
        self.set()
        if mode =='bw':
            self.click += 1

    def show(self):
        total_time = sum(self.time_table.values())
        self.time_table['avg'] = total_time/self.click
        self.time_table['rd'] = 100*self.time_table['rd']/total_time
        self.time_table['fw'] = 100*self.time_table['fw']/total_time
        self.time_table['bw'] = 100*self.time_table['bw']/total_time
        msg  = '{avg:.3f} sec/step (rd {rd:.1f}% | fw {fw:.1f}% | bw {bw:.1f}%)'.format(**self.time_table)
        self.clear()
        return msg

    def clear(self):
        self.time_table = {'rd':0,'fw':0,'bw':0}
        self.click = 0

# Reference : https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/e2e_asr.py#L168
def init_weights(module):
    # Exceptions
    if type(module) == nn.Embedding:
        module.weight.data.normal_(0, 1)
    else:
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
            elif data.dim() in [3,4]:
                # conv weight
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError
def init_gate(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)
    return bias

# Convert Tensor to Figure on tensorboard
def feat_to_fig(feat, spec=False):
    # feat TxD tensor
    data = _save_canvas(feat.numpy(), spec=spec)
    return torch.FloatTensor(data),"HWC"

def _save_canvas(data, meta=None, spec=False):
    sx = 16
    sy = 8
    if spec:
        sx = 24
        sy = 8
    fig, ax = plt.subplots(figsize=(sx, sy))
    if meta is None:
        ax.imshow(data, aspect="auto", origin="lower")
    else:
        ax.bar(meta[0],data[0],tick_label=meta[1],fc=(0, 0, 1, 0.5))
        ax.bar(meta[0],data[1],tick_label=meta[1],fc=(1, 0, 0, 0.5))
    fig.canvas.draw()
    # Note : torch tb add_image takes color as [0,1]
    data = np.array(fig.canvas.renderer._renderer)[:,:,:-1]/255.0 
    plt.close(fig)
    return data

# Reference : https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
def human_format(num):
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '{:3.1f}{}'.format(num, [' ', 'K', 'M', 'G', 'T', 'P'][magnitude])

def cal_er(tokenizer, pred, truth, mode='wer', ctc=False):
    import editdistance as ed
    # Calculate error rate of a batch
    if pred is None:
        return np.nan
    elif len(pred.shape)>=3:
        pred = pred.argmax(dim=-1)
    er = []
    for p,t in zip(pred,truth):
        p = tokenizer.decode(p.tolist(), ignore_repeat=ctc)
        t = tokenizer.decode(t.tolist())
        if mode == 'wer' or mode == 'per':
            p = p.split(' ')
            t = t.split(' ')
        error = 1. if len(t) == 0 else float(ed.eval(p,t))/len(t)
        er.append(error)
    return sum(er)/len(er)


def load_embedding(text_encoder, embedding_filepath):
    with open(embedding_filepath, "r") as f:
        vocab_size, embedding_size = [int(x) for x in f.readline().strip().split()]
        embeddings = np.zeros((text_encoder.vocab_size, embedding_size))

        unk_count = 0

        for line in f:
            vocab, emb = line.strip().split(" ", 1)
            # fasttext's <eos> is </s>
            if vocab == "</s>":
                vocab = "<eos>"

            if text_encoder.token_type == "subword":
                idx = text_encoder.spm.piece_to_id(vocab)
            else:
                # get rid of <eos>
                idx = text_encoder.encode(vocab)[0]

            if idx == text_encoder.unk_idx:
                unk_count += 1
                embeddings[idx] += np.asarray([float(x) for x in emb.split(" ")])
            else:
                # Suppose there is only one (w, v) pair in embedding file
                embeddings[idx] = np.asarray([float(x) for x in emb.split(" ")])

        # Average <unk> vector
        if unk_count != 0:
            embeddings[text_encoder.unk_idx] /= unk_count

        return embeddings

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
