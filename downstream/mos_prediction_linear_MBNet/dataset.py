from pathlib import Path

import random
import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate


class VCC18Dataset(Dataset):
    def __init__(self, dataframe, base_path, idtable="", valid=False):
        self.wav_dir = Path(base_path) / "Converted_speech_of_submitted_systems"
        self.dataframe = dataframe

        if Path.is_file(idtable):
            self.idtable = torch.load(idtable)
            for i, judge_i in enumerate(self.dataframe["JUDGE"]):
                self.dataframe["JUDGE"][i] = self.idtable[judge_i]

        elif not valid:
            self.gen_idtable(idtable)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        wav_name, mean, mos, judge_id = self.dataframe.loc[idx]
        wav_path = self.wav_dir / wav_name
        wav, _ = apply_effects_file(
            wav_path,
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ],
        )

        wav = wav.squeeze(0)
        system_name = wav_name[:3] + wav_name[-8:-4]

        return (
            wav,
            torch.FloatTensor([mean]),
            system_name,
            torch.FloatTensor([mos]),
            judge_id,
        )

    def collate_fn(self, samples):
        wavs, scores, system_names, moss, judge_ids = zip(*samples)
        return (
            wavs,
            torch.stack(scores),
            system_names,
            torch.stack(moss),
            torch.LongTensor(judge_ids),
        )
        
    def gen_idtable(self, idtable_path):
        if idtable_path == '':
            idtable_path = './idtable.pkl'
        self.idtable = {}
        count = 0
        for i, judge_i in enumerate(self.dataframe['JUDGE']):
            if judge_i not in self.idtable.keys():
                self.idtable[judge_i] = count
                count += 1
                self.dataframe['JUDGE'][i] = self.idtable[judge_i]
            else:
                self.dataframe['JUDGE'][i] = self.idtable[judge_i]
        torch.save(self.idtable, idtable_path)


class VCC16Dataset(Dataset):
    def __init__(self, wav_list, base_path):
        self.wav_dir = Path(base_path)
        self.wav_list = wav_list

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_name = self.wav_list[idx]
        wav_path = self.wav_dir / wav_name
        wav, _ = apply_effects_file(
            wav_path,
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ],
        )

        wav = wav.squeeze(0)
        system_name = wav_name.name.split("_")[0]

        return wav, system_name

    def collate_fn(self, samples):
        wavs, system_names = zip(*samples)

        return (
            wavs,
            None,
            system_names,
            None,
            None,
        )


def gen_idtable(self, idtable_path):
    if idtable_path == "":
        idtable_path = "./idtable.pkl"
    self.idtable = {}
    count = 0
    for i, judge_i in enumerate(self.dataframe["JUDGE"]):
        if judge_i not in self.idtable.keys():
            self.idtable[judge_i] = count
            count += 1
            self.dataframe["JUDGE"][i] = self.idtable[judge_i]
        else:
            self.dataframe["JUDGE"][i] = self.idtable[judge_i]
    torch.save(self.idtable, idtable_path)
