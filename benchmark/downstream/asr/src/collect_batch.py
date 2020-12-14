import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

HALF_BATCHSIZE_AUDIO_LEN = 800 # Batch size will be halfed if the longest wavefile surpasses threshold
# Note: Bucketing may cause random sampling to be biased (less sampled for those length > HALF_BATCHSIZE_AUDIO_LEN )
HALF_BATCHSIZE_TEXT_LEN = 150

def collect_audio_batch(batch, audio_transform, mode):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
       e.g. [(file1,txt1),(file2,txt2),...] '''
    
    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]
    # Make sure that batch size is reasonable
    # For each bucket, the first audio must be the longest one
    # But for multi-dataset, this is not the case !!!!
    
    if HALF_BATCHSIZE_AUDIO_LEN < 3500 and mode == 'train':
        first_len = audio_transform(str(batch[0][0])).shape[0]
        if first_len > HALF_BATCHSIZE_AUDIO_LEN:
            batch = batch[::2]
    
    # Read batch
    file, audio_feat, audio_len, text = [],[],[],[]
    with torch.no_grad():
        for index, b in enumerate(batch):
            if type(b[0]) is str:
                file.append(str(b[0]).split('/')[-1].split('.')[0])
                feat = audio_transform(str(b[0]))
            else:
                file.append('dummy')
                feat = audio_transform(str(b[0]))
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(torch.LongTensor(b[1]))
    # Descending audio length within each batch
    audio_len, file, audio_feat, text = zip(*[(feat_len,f_name,feat,txt) \
        for feat_len,f_name,feat,txt in zip(audio_len,file,audio_feat,text)])
    #print(len(audio_feat))
    # Zero-padding
    audio_feat = pad_sequence(audio_feat, batch_first=True)
    text = pad_sequence(text, batch_first=True)
    audio_len = torch.LongTensor(audio_len)
    
    return file, audio_feat, audio_len, text

def collect_text_batch(batch, mode):
    '''Collects a batch of text, should be list of list of int token 
       e.g. [txt1 <list>,txt2 <list>,...] '''

    # Bucketed batch should be [[txt1, txt2,...]]
    if type(batch[0][0]) is list:
        batch = batch[0]
    # Half batch size if input to long
    if len(batch[0])>HALF_BATCHSIZE_TEXT_LEN and mode=='train':
        batch = batch[:len(batch)//2]
    # Read batch
    text = [torch.LongTensor(b) for b in batch]
    # Zero-padding
    text = pad_sequence(text, batch_first=True)
    
    return text