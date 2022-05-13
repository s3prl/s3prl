from math import nan
import torch
import os 
import pandas as pd
from torch.utils.data import Dataset
from tokenizers import Tokenizer
SAMPLE_RATE = 16000
import torchaudio

from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, TimeStretch, Shift, PitchShift, Gain

def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
    return wav.squeeze()

class AtisDataset(Dataset):
    def __init__(self, csv_file, audio_dir, tokenizer, aug_config=None):
        df = pd.read_csv(csv_file)
        ids = df['id'].values
        labels = df['label'].values
        self.audios = []
        self.labels = []
        self.aug_config = aug_config
        for id, label in zip(ids, labels):
            if type(label) is not float:
                audio_file = os.path.join(audio_dir, id+'.wav') 
                if os.path.exists(audio_file):
                    self.audios.append(audio_file)
                    self.labels.append(tokenizer.encode(('<BOS>'+' '+label+' '+'<EOS>')).ids)
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

        return torch.tensor(audio), label
    def collate_fn(self, samples):
        return zip(*samples)
        

if __name__ == '__main__':
    base_path = '/home/daniel094144/data/atis'
    tokenizer = Tokenizer.from_file(os.path.join(base_path,'tokenizer.json'))
    Train_dataset = AtisDataset(os.path.join(base_path,'atis_sv_train.csv'), os.path.join(base_path, 'train'), tokenizer)
    Dev_dataset = AtisDataset(os.path.join(base_path,'atis_sv_dev.csv'), os.path.join(base_path, 'dev'), tokenizer)
    Test_dataset = AtisDataset(os.path.join(base_path,'atis_sv_test.csv'), os.path.join(base_path, 'test'), tokenizer)

    print(Train_dataset[0])



