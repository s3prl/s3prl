from math import nan
import torch
import os 
import pandas as pd
from torch.utils.data import Dataset
from tokenizers import Tokenizer
SAMPLE_RATE = 16000
import torchaudio
from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, TimeStretch, Shift, PitchShift, Gain
import numpy as np

def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
    return wav.squeeze()

BOS_IDX = 2
EOS_IDX = 1
PAD_IDX = 0

class AtisDataset(Dataset):
    def __init__(self, csv_file, audio_dir, tokenizer, aug_config=None, unit_path=None, unit_tokenizer=None):
        df = pd.read_csv(csv_file)
        ids = df['id'].values
        labels = df['label'].values
        self.audios = []
        self.labels = []
        self.unit_tokenizer = unit_tokenizer
        
        if unit_path is not None: 
            self.is_unit = True
        if self.is_unit: 
            self.units = []
        self.aug_config = aug_config
        for id, label in zip(ids, labels):
            if type(label) is not float:
                audio_file = os.path.join(audio_dir, id+'.wav') 
                if os.path.exists(audio_file):
                    self.audios.append(audio_file)
                    self.labels.append(tokenizer.encode(('<BOS>'+' '+label+' '+'<EOS>')).ids)
                    if self.is_unit:
                        self.units.append(os.path.join(unit_path, id+'.wav.code'))

                else: 
                    print(f'{audio_file} is missing')

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        audio_file = self.audios[idx]
        label = self.labels[idx]
        
        audio = reader(audio_file).numpy()  
        if self.aug_config:
            aug_list = []
            for aug, params in self.aug_config.items():
                aug_func = eval(aug)
                aug_list.append(aug_func(**params))

            augment = Compose(aug_list)
            audio = augment(samples=audio, sample_rate=SAMPLE_RATE)  

        if self.is_unit:
            if self.unit_tokenizer is not None: 
                with open(self.units[idx], 'r') as f: 
                    for line in f:
                        unit = list(np.array(self.unit_tokenizer.encode((line)).ids).astype(int) + 3)

            else: 
                unit = list(np.loadtxt(self.units[idx]).astype(int) + 3)
                
            unit = np.array([BOS_IDX] + unit + [EOS_IDX])
            return torch.tensor(audio), label, unit
        else:
            return torch.tensor(audio), label
    def collate_fn(self, samples):
        return zip(*samples)
        

if __name__ == '__main__':
    base_path = '/home/daniel094144/data/atis'
    tokenizer = Tokenizer.from_file(os.path.join(base_path,'tokenizer.json'))
    # Train_dataset = AtisDataset(os.path.join(base_path,'atis_sv_train.csv'), os.path.join(base_path, 'train'), tokenizer)
    Dev_dataset = AtisDataset(os.path.join(base_path,'atis_sv_dev.csv'), os.path.join(base_path, 'dev'), tokenizer, unit_path='/home/daniel094144/data/atis/atis_code_128_IN/code')
    # Test_dataset = AtisDataset(os.path.join(base_path,'atis_sv_test.csv'), os.path.join(base_path, 'test'), tokenizer)

    print(Dev_dataset[0])



