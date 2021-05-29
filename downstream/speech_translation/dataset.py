import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os 

import csv
import torchaudio
import sentencepiece as spm
from tqdm.auto import tqdm

class COVOST2Dataset(Dataset):

    SAMPLE_RATE = 48000

    def __init__(self, src_lang, tgt_lang, split, root_dir, tsv_dir, tokenizer, max_length = -1, wav_max_length = -1, sample_rate = 16000, remove_blank = True, target='translation'):
        super().__init__()
        
        self.clip_dir = f'{root_dir}/{src_lang}/clips/'
        self.data = []
        self.tokenizer = tokenizer

        self.resampler = torchaudio.transforms.Resample(
            orig_freq = self.SAMPLE_RATE,
            new_freq = sample_rate,
        )
        blank_count = 0
        tsv_file = f'{tsv_dir}/covost_v2.{src_lang}_{tgt_lang}.{split}.tsv'
        with open(tsv_file, 'r') as f:
            for line in csv.DictReader(f, delimiter='\t'):
                path = line['path']
                tgt = line[target].strip()
                if remove_blank and tgt == '':
                    blank_count += 1
                    continue
                self.data.append((path, tgt))
        self.max_length = max_length
        self.wav_max_length = wav_max_length

        if remove_blank:
            tqdm.write(f'remove {blank_count} blank translation from split {split}')


    def __getitem__(self, idx):
        wav = self._load_wav(self.data[idx][0])
        label = torch.LongTensor(self.tokenizer.encode(self.data[idx][1]))
        if self.max_length > 0:
            if len(label) > self.max_length:
                tqdm.write(f'label length too long ({len(label)}), cut to {self.max_length}')
            label = label[:self.max_length]
        if self.wav_max_length > 0:
            if len(wav) > self.wav_max_length:
                tqdm.write(f'wav length too long ({len(wav)}), cut to {self.wav_max_length}')
            wav = wav[:self.wav_max_length]
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
