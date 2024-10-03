import os
import json
import random

from tqdm import tqdm
import torch

from pathlib import Path
from librosa.util import find_files
from joblib.parallel import Parallel, delayed
from torch.utils.data import Dataset
from torchaudio.sox_effects import apply_effects_file

EFFECTS = [
["channels", "1"],
["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

import sys
import argparse
import torch.nn.functional as F
sys.path.append("/mnt/andy9_liu/work/fairseq")
import fairseq

def get_dinosr_model(code_path, ckpt_path, device="cup"):
    fairseq.utils.import_user_module(argparse.Namespace(user_dir=code_path))
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    model.eval()
    model.to(device)
    model.feature_grad_mult = 0
    return model

def single_instance_forward(model, wav):
    # wav shape: (1, wav_length)
    x = model(source=wav, mask=False, features_only=True)['x']
    pred = model.heads[-1](x).float()
    unit = F.log_softmax(pred, dim=-1)
    unit = torch.argmax(unit, dim=-1)
    return unit.squeeze().tolist()

def extract_unit_from_wav_normalized(model, device, wav):
    """
    Using the layer normalization to preprocess the wav, following the fairseq implementation:
    https://github.com/facebookresearch/fairseq/blob/920a548ca770fb1a951f7f4289b4d3a0c1bc226f/examples/textless_nlp/gslm/speech2unit/pretrained/hubert_feature_reader.py#L57
    """
    wav = F.layer_norm(wav.float(), wav.shape)
    wav = wav.view(1, -1)
    unit = single_instance_forward(model, wav.to(device))
    return unit

# Voxceleb 2 Speaker verification
class SpeakerVerifi_train(Dataset):
    def __init__(self, vad_config, key_list, file_path, meta_data, max_timestep=None, n_jobs=12, load_discrete=True):
        self.roots = file_path
        self.root_key = key_list
        self.max_timestep = max_timestep
        self.vad_c = vad_config 
        self.dataset = []
        self.all_speakers = []

        for index in range(len(self.root_key)):
            cache_path = Path(os.path.dirname(__file__)) / '.wav_lengths' / f'{self.root_key[index]}_length.pt'
            cache_path.parent.mkdir(exist_ok=True)
            root = Path(self.roots[index])

            if not cache_path.is_file():
                def trimmed_length(path):
                    wav_sample, _ = apply_effects_file(path, EFFECTS)
                    wav_sample = wav_sample.squeeze(0)
                    length = wav_sample.shape[0]
                    return length

                wav_paths = find_files(root)
                wav_lengths = Parallel(n_jobs=n_jobs)(delayed(trimmed_length)(path) for path in tqdm.tqdm(wav_paths, desc="Preprocessing"))
                wav_tags = [Path(path).parts[-3:] for path in wav_paths]
                torch.save([wav_tags, wav_lengths], str(cache_path))
            else:
                wav_tags, wav_lengths = torch.load(str(cache_path))
                wav_paths = [root.joinpath(*tag) for tag in wav_tags]

            speaker_dirs = ([f.stem for f in root.iterdir() if f.is_dir()])
            self.all_speakers.extend(speaker_dirs)
            for path, length in zip(wav_paths, wav_lengths):
                if length > self.vad_c['min_sec']:
                    self.dataset.append(path)

        self.all_speakers.sort()
        self.speaker_num = len(self.all_speakers)
        self.load_discrete = load_discrete
        if self.load_discrete:
            self.preprocess_discrete_dinosr(redo=True)

    def preprocess_discrete_dinosr(self, redo=False):
        # Create cache directory if it doesn't exist
        cache_dir = Path("/mnt/andy9_liu/dataset/preprocessed_cache")
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"discrete_units_voxceleb1_dinosr_train.json"
        
        # Try to load from cache
        if not redo and cache_path.exists():
            print(f"Loading cached preprocessed data from {cache_path}")
            with open(cache_path, 'r') as f:
                self.dataset_discrete = json.load(f)
            return
    
        code_path = '/mnt/andy9_liu/fairseq/examples/dinosr'
        ckpt_path = '/mnt/andy9_liu/fairseq/examples/dinosr/dinosr.ckpt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dinosr_model = get_dinosr_model(code_path, ckpt_path, device)
       
        self.dataset_discrete = {}
        for path in tqdm(self.dataset):
            wav, _ = apply_effects_file(str(path), EFFECTS)
            unit = extract_unit_from_wav_normalized(dinosr_model, device, wav)
            if len(unit) > 1095: # max_seq_length of mlm
                start = random.randint(0, len(unit) - 1095)
                unit = unit[start : start + 1095]
            unit_str = '_'.join(map(str, unit))
            self.dataset_discrete[str(path)] = unit_str
            
        # Save results to cache
        print(f"Saving preprocessed data to cache: {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(self.dataset_discrete, f)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        path = self.dataset[idx]
        tags = Path(path).parts[-3:]
        utterance_id = "-".join(tags).replace(".wav", "")
        label = self.all_speakers.index(tags[0])
        
        if self.load_discrete:
            return self.dataset_discrete[str(path)], utterance_id, label
        else:
            wav, _ = apply_effects_file(str(path), EFFECTS)
            wav = wav.squeeze(0)
            length = wav.shape[0]
            
            if self.max_timestep != None:
                if length > self.max_timestep:
                    start = random.randint(0, int(length - self.max_timestep))
                    wav = wav[start : start + self.max_timestep]
            return wav.numpy(), utterance_id, label
        
    def collate_fn(self, samples):
        return zip(*samples)


class SpeakerVerifi_test(Dataset):
    def __init__(self, vad_config, file_path, meta_data, load_discrete=True):
        self.root = file_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        self.vad_c = vad_config 
        self.dataset = self.necessary_dict['spk_paths']
        self.pair_table = self.necessary_dict['pair_table']
        self.load_discrete = load_discrete
        if self.load_discrete:
            self.split = "test" if "test" in self.meta_data else "dev"
            self.preprocess_discrete_dinosr(redo=True)

        
    def processing(self):
        pair_table = []
        spk_paths = set()
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1= os.path.join(self.root, list_pair[1])
            pair_2= os.path.join(self.root, list_pair[2])
            spk_paths.add(pair_1)
            spk_paths.add(pair_2)
            one_pair = [list_pair[0],pair_1,pair_2 ]
            pair_table.append(one_pair)
        return {
            "spk_paths": list(spk_paths),
            "total_spk_num": None,
            "pair_table": pair_table
        }
        
    def preprocess_discrete_dinosr(self, redo=False):
        # Create cache directory if it doesn't exist
        cache_dir = Path("/mnt/andy9_liu/dataset/preprocessed_cache")
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"discrete_units_voxceleb1_dinosr_{self.split}.json"
        
        # Try to load from cache
        if not redo and cache_path.exists():
            print(f"Loading cached preprocessed data from {cache_path}")
            with open(cache_path, 'r') as f:
                self.dataset_discrete = json.load(f)
            return

        code_path = '/mnt/andy9_liu/fairseq/examples/dinosr'
        ckpt_path = '/mnt/andy9_liu/fairseq/examples/dinosr/dinosr.ckpt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dinosr_model = get_dinosr_model(code_path, ckpt_path, device)
       
        self.dataset_discrete = {}
        for path in tqdm(self.dataset):
            wav, _ = apply_effects_file(str(path), EFFECTS)
            unit = extract_unit_from_wav_normalized(dinosr_model, device, wav)
            if len(unit) > 1095: # max_seq_length of mlm
                unit = unit[:1095]
            unit_str = '_'.join(map(str, unit))
            self.dataset_discrete[str(path)] = unit_str
            
        # Save results to cache
        print(f"Saving preprocessed data to cache: {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(self.dataset_discrete, f)

    def __len__(self):
        return len(self.necessary_dict['spk_paths'])

    def __getitem__(self, idx):
        x_path = self.dataset[idx]

        if self.load_discrete:
            return self.dataset_discrete[str(x_path)], x_path
        else:
            x_name = x_path
            wav, _ = apply_effects_file(x_path, EFFECTS)
            wav = wav.squeeze(0)
            return wav.numpy(), x_name

    def collate_fn(self, data_sample):
        wavs, x_names = zip(*data_sample)
        return wavs, x_names
