import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import accumulate
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file, apply_effects_tensor


class QUESST14Trainset(Dataset):
    """QUESST 2014 training dataset."""

    def __init__(self, split, **kwargs):
        dataset_root = Path(kwargs["quesst2014_root"])
        scoring_root = dataset_root / "scoring"
        split_root = scoring_root / f"groundtruth_quesst14_{split}"

        # parse infos
        query2positives = parse_rttm(split_root / f"quesst14_{split}.rttm")
        audio_names = parse_lst(scoring_root / "language_key_utterances.lst")
        query_names = parse_lst(scoring_root / f"language_key_{split}.lst")
        print(f"[QUESST2014] # of audios: {len(audio_names)}")
        print(f"[QUESST2014] # of queries: {len(query_names)}")

        # find complement set
        audio_set = set(audio_names)
        query2negatives = {
            query_name: list(
                audio_set
                - set(
                    query2positives[query_name] if query_name in query2positives else []
                )
            )
            for query_name in query_names
        }

        # form positive & negative pairs
        positive_pairs = [
            (query_name, audio_name)
            for query_name in query_names
            for audio_name in set(query2positives[query_name]) & audio_set
        ]
        negative_pairs = [
            (query_name, list(negative_audio_set))
            for query_name, negative_audio_set in query2negatives.items()
        ]
        print(f"[QUESST2014] # of positive pairs: {len(positive_pairs)}")
        print(f"[QUESST2014] # of negative pairs: {len(negative_pairs)}")

        self.audio_root = dataset_root / "Audio"
        self.query_root = dataset_root / f"{split}_queries"
        self.max_dur = 3.0
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs

    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs)

    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):  # positive pair
            query_name, audio_name = self.positive_pairs[idx]
        else:  # negative pair
            query_name, audio_names = self.negative_pairs[
                idx - len(self.positive_pairs)
            ]
            audio_name = random.sample(audio_names, 1)[0]

        query_path = (self.query_root / query_name).with_suffix(".wav")
        audio_path = (self.audio_root / audio_name).with_suffix(".wav")

        query_tensor = path2tensor(query_path)
        audio_tensor = path2tensor(audio_path)

        query_segment = crop_segment(query_tensor, self.max_dur)
        audio_segments = unfold_segments(audio_tensor, self.max_dur)
        label = torch.LongTensor([1 if idx < len(self.positive_pairs) else -1])

        return query_segment, audio_segments, label

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        query_segments, segments_list, labels = zip(*samples)
        flattened = [segment for segments in segments_list for segment in segments]
        lengths = [len(segments) for segments in segments_list]
        prefix_sums = list(accumulate(lengths, initial=0))
        return list(query_segments) + flattened, (prefix_sums, labels)

    @property
    def sample_weights(self):
        """Sample weights to balance positive and negative data."""
        n_pos = len(self.positive_pairs)
        n_neg = len(self.negative_pairs)
        return [1 / n_pos] * n_pos + [1 / n_neg] * n_neg


def parse_rttm(rttm_path):
    """Parse audio and query pairs from *.rttm."""

    # e.g. "LEXEME quesst14_12345 ... quesst14_dev_123 ..."
    pattern = re.compile(r"LEXEME\s+(quesst14_[0-9]+).*?(quesst14_(dev|eval)_[0-9]+)")

    query2audios = defaultdict(list)
    with open(rttm_path) as fd:
        for line in fd:
            match = pattern.match(line)
            if match is None:
                continue
            query2audios[match.group(2)].append(match.group(1))

    return query2audios


def parse_lst(lst_path):
    """Extract audio names of nnenglish."""
    audio_names = []

    with open(lst_path) as fd:
        for line in fd:
            audio_path, lang = tuple(line.strip().split())
            if lang != "nnenglish":
                continue
            audio_name = Path(audio_path).with_suffix("").name
            audio_names.append(audio_name)

    return audio_names


def path2tensor(filepath):
    tensor, _ = apply_effects_file(
        str(filepath),
        [
            ["channels", "1"],
            ["rate", "16000"],
            ["norm"],
        ],
    )
    return tensor.squeeze(0)


def crop_segment(tensor, tgt_dur, sample_rate=16000):
    src_dur = len(tensor) / sample_rate
    random_shift = random.uniform(0, src_dur - tgt_dur)
    audio_tensor, _ = apply_effects_tensor(
        tensor.unsqueeze(0),
        sample_rate,
        [
            ["pad", f"{tgt_dur}", f"{tgt_dur}"],
            [
                "trim",
                f"{tgt_dur + random_shift}",
                f"{tgt_dur}",
            ],
        ],
    )
    return audio_tensor.squeeze(0)


def unfold_segments(tensor, tgt_dur, sample_rate=16000):
    seg_len = int(tgt_dur * sample_rate)
    src_len = len(tensor)
    hop_len = seg_len // 4
    tgt_len = seg_len if src_len <= seg_len else (src_len // hop_len + 1) * hop_len

    pad_len = tgt_len - src_len
    front_pad_len = random.randint(0, pad_len)
    tail_pad_len = pad_len - front_pad_len

    padded_tensor = torch.cat(
        [torch.zeros(front_pad_len), tensor, torch.zeros(tail_pad_len)]
    )
    segments = padded_tensor.unfold(0, seg_len, hop_len).unbind(0)

    return segments
