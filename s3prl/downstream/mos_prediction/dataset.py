from pathlib import Path
import os

import random
import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate


class VCC18SegmentalDataset(Dataset):
    def __init__(self, dataframe, base_path, idtable = '', valid = False):
        self.base_path = Path(base_path)
        self.dataframe = dataframe
        self.segments_durations = 1
        if Path.is_file(idtable):
            self.idtable = torch.load(idtable)
            for i, judge_i in enumerate(self.dataframe['JUDGE']):
                self.dataframe['JUDGE'][i] = self.idtable[judge_i]

        elif not valid:
            self.gen_idtable(idtable)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        wav_name, mean, mos, judge_id = self.dataframe.loc[idx]
        wav_path = self.base_path / "Converted_speech_of_submitted_systems" / wav_name
        wav, _ = apply_effects_file(
            str(wav_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ],
        )

        wav = wav.view(-1)
        wav_segments = unfold_segments(wav, self.segments_durations)
        system_name = wav_name[:3] + wav_name[-8:-4]

        return wav_segments, mean, system_name, mos, judge_id

    def collate_fn(self, samples):
        wavs_segments, means, system_names, moss, judge_ids = zip(*samples)
        flattened_wavs_segments = [
            wav_segment
            for wav_segments in wavs_segments
            for wav_segment in wav_segments
        ]
        wav_segments_lengths = [len(wav_segments) for wav_segments in wavs_segments]
        prefix_sums = list(accumulate(wav_segments_lengths, initial=0))
        segment_judge_ids = []
        for i in range(len(prefix_sums)-1):
            segment_judge_ids.extend([judge_ids[i]] * (prefix_sums[i+1]-prefix_sums[i]))
        
        return (
            torch.stack(flattened_wavs_segments),
            prefix_sums,
            torch.FloatTensor(means),
            system_names,
            torch.FloatTensor(moss), 
            torch.LongTensor(segment_judge_ids)
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


class VCC16SegmentalDataset(Dataset):
    def __init__(self, wav_list, base_path):
        self.wav_dir = Path(base_path)
        self.wav_list = wav_list
        self.segments_durations = 1

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_name = self.wav_list[idx]
        wav_path = self.wav_dir / wav_name
        wav, _ = apply_effects_file(
            str(wav_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ],
        )

        wav = wav.view(-1)
        wav_segments = unfold_segments(wav, self.segments_durations)
        system_name = wav_name.name.split("_")[0]

        return wav_segments, system_name

    def collate_fn(self, samples):
        wavs_segments, system_names = zip(*samples)
        flattened_wavs_segments = [
            wav_segment
            for wav_segments in wavs_segments
            for wav_segment in wav_segments
        ]
        wav_segments_lengths = [len(wav_segments) for wav_segments in wavs_segments]
        prefix_sums = list(accumulate(wav_segments_lengths, initial=0))

        return (
            torch.stack(flattened_wavs_segments),
            prefix_sums,
            None,
            system_names,
            None, 
            None,
        )


def unfold_segments(tensor, tgt_duration, sample_rate=16000):
    seg_lengths = int(tgt_duration * sample_rate)
    src_lengths = len(tensor)
    step = seg_lengths // 2
    tgt_lengths = (
        seg_lengths if src_lengths <= seg_lengths else (src_lengths // step + 1) * step
    )

    pad_lengths = tgt_lengths - src_lengths
    padded_tensor = torch.cat([tensor, torch.zeros(pad_lengths)])
    segments = padded_tensor.unfold(0, seg_lengths, step).unbind(0)

    return segments
