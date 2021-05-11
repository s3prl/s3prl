import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os 

import csv
import torchaudio
import sentencepiece as spm


class COVOST2Dataset(Dataset):

    SAMPLE_RATE = 48000

    def __init__(self, src_lang, tgt_lang, split, root_dir, tsv_dir, tokenizer, max_length = -1, sample_rate = 16000):
        super().__init__()
        
        self.clip_dir = f'{root_dir}/{src_lang}/clips/'
        self.data = []
        self.tokenizer = tokenizer

        self.resampler = torchaudio.transforms.Resample(
            orig_freq = self.SAMPLE_RATE,
            new_freq = sample_rate,
        )

        tsv_file = f'{tsv_dir}/covost_v2.{src_lang}_{tgt_lang}.{split}.tsv'
        with open(tsv_file, 'r') as f:
            for line in csv.DictReader(f, delimiter='\t'):
                self.data.append((line['path'], line['translation']))
        self.max_length = max_length

    def __getitem__(self, idx):
        wav = self._load_wav(self.data[idx][0])
        label = torch.LongTensor(self.tokenizer.encode(self.data[idx][1]))
        if self.max_length > 0:
            label = label[:self.max_length]
        return wav, label

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels

    def _load_wav(self, path):
        wav, sr = torchaudio.load(os.path.join(self.clip_dir, path))
        assert sr == self.SAMPLE_RATE
        wav = self.resampler(wav)
        return wav.view(-1)

if __name__ == '__main__':

    src_lang = 'fr'
    tgt_lang = 'en'
    split = 'test'

    root_dir = '/livingrooms/public/CoVoST2/cv-corpus-6.1-2020-12-11/'
    tsv_dir = '/livingrooms/public/CoVoST2/tsv/'
    tokenizer_path = '/home/sean/s3prl/downstream/speech_translation/spm_fr_en_8000.model'

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

    dataset = STDataset(src_lang, tgt_lang, split, root_dir, tsv_dir, tokenizer)

    print(dataset[0], )
