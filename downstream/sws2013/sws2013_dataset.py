import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file, apply_effects_tensor
from tqdm import tqdm


class SWS2013Dataset(Dataset):
    """SWS 2013 dataset."""

    def __init__(self, split, **kwargs):
        assert split in ["dev", "eval"]

        dataset_root = Path(kwargs["sws2013_root"])
        split_root = Path(kwargs["sws2013_scoring_root"]) / f"sws2013_{split}"

        # parse infos
        audio2dur = parse_ecf(split_root / "sws2013.ecf.xml")
        query2audios = parse_rttm(split_root / f"sws2013_{split}.rttm")
        query2tensors = find_queries(dataset_root / f"{split}_queries")
        print(f"[SWS2013] # of audios: {len(audio2dur)}")
        print(f"[SWS2013] # of queries: {len(query2tensors)}")

        # find complement set
        all_audio_set = set(audio2dur.keys())
        query2audio_set = {
            query: set(audio_info["audio"] for audio_info in audio_infos)
            for query, audio_infos in query2audios.items()
        }
        query2audio_compl_set = {
            query: all_audio_set - audio_set
            for query, audio_set in query2audio_set.items()
        }

        # form positive & negative pairs
        positive_pairs, negative_pairs = [], []
        for query, tensors in query2tensors.items():
            for query_tensor in tensors:
                negative_pairs.append(
                    {
                        "query_tensor": query_tensor,
                        "audio_set": query2audio_compl_set[query]
                        if query in query2audio_compl_set
                        else all_audio_set,
                    }
                )
                if query not in query2audios:
                    continue
                for audio_info in query2audios[query]:
                    positive_pairs.append(
                        {
                            "query_tensor": query_tensor,
                            "audio": audio_info["audio"],
                            "offset": audio_info["offset"],
                            "duration": audio_info["duration"],
                        }
                    )

        print(f"[SWS2013] # of positive pairs: {len(positive_pairs)}")

        self.audio_dir = dataset_root / "Audio"
        self.audio2dur = audio2dur
        self.max_dur = 3.0
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs

    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs)

    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):  # positive pair
            pair = self.positive_pairs[idx]
            audio_path = (self.audio_dir / pair["audio"]).with_suffix(".wav")
            audio_tensor = path2segment(
                audio_path, pair["duration"], self.max_dur, pair["offset"]
            )
        else:  # negative pair
            pair = self.negative_pairs[idx - len(self.positive_pairs)]
            sample_audio = random.sample(pair["audio_set"], 1)[0]
            audio_dur = self.audio2dur[sample_audio]
            audio_path = (self.audio_dir / sample_audio).with_suffix(".wav")
            audio_tensor = path2segment(audio_path, audio_dur, self.max_dur, 0.0)

        audio_tensor = audio_tensor.squeeze(0)
        query_tensor = tensor2segment(pair["query_tensor"], self.max_dur)
        label = torch.LongTensor([1 if idx < len(self.positive_pairs) else -1])

        return audio_tensor, query_tensor, label

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        audio_tensors, query_tensors, labels = zip(*samples)
        return audio_tensors + query_tensors, labels

    @property
    def sample_weights(self):
        """Sample weights to balance positive and negative data."""
        n_pos = len(self.positive_pairs)
        n_neg = len(self.negative_pairs)
        return [1 / n_pos] * n_pos + [1 / n_neg] * n_neg


def parse_rttm(rttm_path):
    """Parse audio and query pairs from *.rttm."""

    # e.g. "LEXEME sws2013_12345 ... 3.50 1.00 sws2013_dev_123 ..."
    pattern = re.compile(
        r"LEXEME\s+(sws2013_[0-9]+).*?([0-9]\.[0-9]+)\s+([0-9]\.[0-9]+)"
        r"\s+(sws2013_(dev|eval)_[0-9]+)"
    )

    query2audios = defaultdict(list)
    with open(rttm_path) as fd:
        for line in fd:
            match = pattern.match(line)
            if match is None:
                continue
            query2audios[match.group(4)].append(
                {
                    "audio": match.group(1),
                    "offset": float(match.group(2)),
                    "duration": float(match.group(3)),
                }
            )

    return query2audios


def parse_ecf(ecf_path):
    """Find audios from sws2013.ecf.xml."""

    root = ET.parse(str(ecf_path)).getroot()

    audio2dur = {}
    for excerpt in root.findall("excerpt"):
        audio_name = (
            excerpt.attrib["audio_filename"].replace("Audio/", "").replace(".wav", "")
        )
        duration = float(excerpt.attrib["dur"])
        audio2dur[audio_name] = duration

    return audio2dur


def find_queries(query_dir_path):
    """Find all queries under sws2013_dev & sws2013_eval."""

    # e.g. "sws2013_dev_123.wav" or "sws2013_dev_123_01.wav" -> "sws2013_dev_123"
    pattern = re.compile(r"(_[0-9]{2})?\.wav")

    query2tensors = defaultdict(list)
    for query_path in tqdm(
        list(query_dir_path.glob("*.wav")), ncols=0, desc="Load queries"
    ):
        query_name = pattern.sub("", query_path.name)
        wav_tensor, sample_rate = apply_effects_file(
            str(query_path), [["channels", "1"], ["rate", "16000"], ["norm"]]
        )
        trimmed, _ = apply_effects_tensor(
            wav_tensor,
            sample_rate,
            [
                ["vad", "-T", "0.25", "-p", "0.1"],
                ["reverse"],
                ["vad", "-T", "0.25", "-p", "0.1"],
                ["reverse"],
            ],
        )
        wav_tensor = trimmed if trimmed.size(1) >= (sample_rate * 0.5) else wav_tensor
        wav_tensor = wav_tensor.squeeze(0)
        query2tensors[query_name].append(wav_tensor)

    return query2tensors


def path2segment(filepath, src_dur, tgt_dur, offset):
    random_shift = random.uniform(0, src_dur - tgt_dur)
    audio_tensor, _ = apply_effects_file(
        str(filepath),
        [
            ["channels", "1"],
            ["rate", "16000"],
            ["norm"],
            ["pad", f"{tgt_dur}", f"{tgt_dur}"],
            [
                "trim",
                f"{tgt_dur + offset + random_shift}",
                f"{tgt_dur}",
            ],
        ],
    )
    return audio_tensor


def tensor2segment(tensor, tgt_dur, sample_rate=16000):
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
