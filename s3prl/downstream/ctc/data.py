import torch
import torchaudio
from functools import partial
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

EVAL_BATCH_SIZE = 1


def collect_audio_batch(batch, split, half_batch_size_wav_len=300000):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
       e.g. [(file1,txt1),(file2,txt2),...]
    '''
    def audio_reader(filepath):
        wav, sample_rate = torchaudio.load(filepath)
        return wav.reshape(-1)

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]

    # Make sure that batch size is reasonable
    first_len = audio_reader(str(batch[0][0])).size(0)
    if split == 'train':
        if first_len > half_batch_size_wav_len and len(batch) > 1:
            batch = batch[:len(batch)//2]

    # Read batch
    file, audio_feat, audio_len, text = [], [], [], []
    with torch.no_grad():
        for b in batch:
            file.append(str(b[0]).split('/')[-1].split('.')[0])
            feat = audio_reader(str(b[0])).numpy()
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(torch.LongTensor(b[1]).numpy())

    # Descending audio length within each batch
    audio_len, file, audio_feat, text = zip(*[(feat_len, f_name, feat, txt)
                                              for feat_len, f_name, feat, txt in sorted(zip(audio_len, file, audio_feat, text), reverse=True, key=lambda x:x[0])])

    return audio_feat, text, file


def create_dataset(split, tokenizer, name, bucketing, batch_size, **kwargs):
    ''' Interface for creating all kinds of dataset'''

    # Recognize corpus
    if name.lower() == "librispeech":
        from .corpus.librispeech import LibriDataset as Dataset
    elif name.lower() == "snips":
        from .corpus.snips import SnipsDataset as Dataset
    elif name.lower() == 'libriphone':
        from .corpus.libriphone import LibriPhoneDataset as Dataset
    elif name.lower() in {"common_voice", "sbcsae"}:
        from .corpus.common_voice import CommonVoiceDataset as Dataset
    else:
        raise NotImplementedError

    if split == 'train':
        kwargs["ratio"] = 1.0
        kwargs["offset"] = 0
        loader_bs = 1 if bucketing else batch_size
        bucket_size = batch_size if bucketing else 1
        dataset = Dataset(kwargs['train'], tokenizer, bucket_size, **kwargs)
    else:
        loader_bs = EVAL_BATCH_SIZE
        dataset = Dataset(kwargs[split], tokenizer, 1, **kwargs)

    return dataset, loader_bs


def load_dataset(split, tokenizer, corpus):
    ''' Prepare dataloader for training/validation'''
    num_workers = corpus.pop('num_workers', 12)
    dataset, loader_bs = create_dataset(split, tokenizer, num_workers=num_workers, **corpus)
    collate_fn = partial(collect_audio_batch, split=split)
    if split == 'train':
        sampler = DistributedSampler(dataset) if is_initialized() else None
        dataloader = DataLoader(dataset, batch_size=loader_bs, shuffle=(sampler is None),
                                sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=loader_bs, shuffle=False,
                                collate_fn=collate_fn, num_workers=num_workers)
    return dataloader
