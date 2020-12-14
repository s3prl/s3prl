import torch
import numpy as np
from functools import partial
from src.text import load_text_encoder
from src.audio import create_transform
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from os.path import join
from src.collect_batch import collect_audio_batch, collect_text_batch

def create_dataset(tokenizer, ascending, name, path, bucketing, batch_size, 
                   train_split=None, dev_split=None, test_split=None, read_audio=False):
    ''' Interface for creating all kinds of dataset'''

    # Recognize corpus
    if name.lower() == 'librispeech':
        from corpus.preprocess_librispeech import LibriDataset as Dataset
    elif name.lower() == 'dlhlp':
        from corpus.preprocess_dlhlp import DLHLPDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    if train_split is not None:
        # Training mode
        mode = 'train'
        tr_loader_bs = 1 if bucketing and (not ascending) else batch_size
        bucket_size = batch_size if bucketing and (not ascending) else 1 # Ascending without bucketing
        
        if type(dev_split[0]) is not list:
            dv_set = Dataset(path,dev_split,tokenizer, 1, read_audio=read_audio) # Do not use bucketing for dev set
            dv_len = len(dv_set)
        else:
            dv_set = []
            for ds in dev_split:
                dev_dir = ''
                if ds[0].lower() == 'librispeech':
                    dev_dir = join(path, 'LibriSpeech')
                    from corpus.preprocess_librispeech import LibriDataset as DevDataset
                else:
                    raise NotImplementedError(ds[0])
                dv_set.append(DevDataset(dev_dir,ds,tokenizer, 1))
            dv_len = sum([len(s) for s in dv_set])
        
        if path[-4:].lower() != name[-4:].lower():
            tr_dir = join(path, name)
        else:
            tr_dir = path
        
        tr_set = Dataset(tr_dir,train_split,tokenizer, bucket_size, 
                    ascending=ascending, 
                    read_audio=read_audio)
        # Messages to show
        msg_list = _data_msg(name,path,train_split.__str__(),len(tr_set),
                             dev_split.__str__(),dv_len,batch_size,bucketing)

        return tr_set, dv_set, tr_loader_bs, batch_size, mode, msg_list
    else:
        # Testing model
        mode = 'test'
        if path[-4:].lower() != name[-4:].lower():
            tt_dir = join(path, name)
        else:
            tt_dir = path
        
        bucket_size = 1
        if type(dev_split[0]) is list: dev_split = dev_split[0]
        
        dv_set = Dataset(tt_dir,dev_split,tokenizer, bucket_size, read_audio=read_audio) # Do not use bucketing for dev set
        tt_set = Dataset(tt_dir,test_split,tokenizer, bucket_size, read_audio=read_audio) # Do not use bucketing for test set
        # Messages to show
        msg_list = _data_msg(name,tt_dir,dev_split.__str__(),len(dv_set),
                             test_split.__str__(),len(tt_set),batch_size,False)
        msg_list = [m.replace('Dev','Test').replace('Train','Dev') for m in msg_list]
        return dv_set, tt_set, batch_size, batch_size, mode, msg_list

def create_textset(tokenizer, train_split, dev_split, name, path, bucketing, batch_size):
    ''' Interface for creating all kinds of text dataset'''
    msg_list = []

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.preprocess_librispeech import LibriTextDataset as Dataset
    elif name.lower() == 'dlhlp':
        from corpus.preprocess_dlhlp import DLHLPTextDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    bucket_size = batch_size if bucketing else 1
    tr_loader_bs = 1 if bucketing else batch_size
    dv_set = Dataset(path,dev_split,tokenizer, 1) # Do not use bucketing for dev set
    tr_set = Dataset(path,train_split,tokenizer, bucket_size)
    
    # Messages to show
    msg_list = _data_msg(name,path,train_split.__str__(),len(tr_set),
                         dev_split.__str__(),len(dv_set),batch_size,bucketing)

    return tr_set, dv_set, tr_loader_bs, batch_size, msg_list


def load_dataset(n_jobs, use_gpu, pin_memory, ascending, corpus, audio, text):
    ''' Prepare dataloader for training/validation'''
    # Audio feature extractor
    '''convert to mel-spectrogram'''
    audio_transform_tr, feat_dim = create_transform(audio.copy(), 'train')
    audio_transform_dv, feat_dim = create_transform(audio.copy(), 'dev')

    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset (in testing mode, tr_set=dv_set, dv_set=tt_set)
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, mode, data_msg = create_dataset(tokenizer,ascending,**corpus)
    
    # Collect function
    collect_tr = partial(collect_audio_batch, audio_transform=audio_transform_tr, mode=mode)
    collect_dv = partial(collect_audio_batch, audio_transform=audio_transform_dv, mode='test')
    
    # Shuffle/drop applied to training set only
    shuffle = (mode=='train' and not ascending)
    drop_last = shuffle
    # Create data loader

    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=shuffle, drop_last=drop_last, collate_fn=collect_tr,
                        num_workers=n_jobs, pin_memory=use_gpu)
    
    if type(dv_set) is list:
        _tmp_set = []
        for ds in dv_set:
            _tmp_set.append(DataLoader(ds, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=n_jobs, pin_memory=pin_memory))
        dv_set = _tmp_set
    else:
        dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=n_jobs, pin_memory=pin_memory)
    
    # Messages to show
    data_msg.append('I/O spec.  | Audio Feature = {}\t| Feature Dim = {}\t| Token Type = {}\t| Vocab Size = {}'\
                    .format(audio['feat_type'],feat_dim,tokenizer.token_type,tokenizer.vocab_size))
    return tr_set, dv_set, feat_dim, tokenizer.vocab_size, tokenizer, data_msg

def load_textset(n_jobs, use_gpu, pin_memory, corpus, text):
    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, data_msg = create_textset(tokenizer,**corpus)
    collect_tr = partial(collect_text_batch,mode='train')
    collect_dv = partial(collect_text_batch,mode='dev')
    # Dataloader (Text data stored in RAM, no need num_workers)
    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=True, drop_last=True, collate_fn=collect_tr,
                        num_workers=0, pin_memory=use_gpu)
    dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=0, pin_memory=pin_memory)

    # Messages to show
    data_msg.append('I/O spec.  | Token type = {}\t| Vocab size = {}'\
                    .format(tokenizer.token_type,tokenizer.vocab_size))

    return tr_set, dv_set, tokenizer.vocab_size, tokenizer, data_msg


def _data_msg(name,path,train_split,tr_set,dev_split,dv_set,batch_size,bucketing):
    ''' List msg for verbose function '''
    msg_list = []
    msg_list.append('Data spec. | Corpus = {} (from {})'.format(name,path))
    msg_list.append('           | Train sets = {}\t| Number of utts = {}'.format(train_split,tr_set))
    msg_list.append('           | Dev sets = {}\t| Number of utts = {}'.format(dev_split,dv_set))
    msg_list.append('           | Batch size = {}\t\t| Bucketing = {}'.format(batch_size,bucketing))
    return msg_list
