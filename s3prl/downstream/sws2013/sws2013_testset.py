import re
import xml.etree.ElementTree as ET
from pathlib import Path

from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file


class SWS2013Testset(Dataset):
    """SWS 2013 testset."""

    def __init__(self, split, **kwargs):
        assert split in ["dev", "eval"]

        scoring_root = Path(kwargs["sws2013_scoring_root"])
        audio_names = parse_ecf(scoring_root / f"sws2013_{split}" / "sws2013.ecf.xml")
        query_names = parse_tlist(
            scoring_root / f"sws2013_{split}" / f"sws2013_{split}.tlist.xml"
        )

        self.dataset_root = Path(kwargs["sws2013_root"])
        self.split = split
        self.n_queries = len(query_names)
        self.n_docs = len(audio_names)
        self.data = query_names + audio_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_name = self.data[idx]
        audio_path = (
            (self.dataset_root / f"{self.split}_queries" / audio_name)
            if idx < self.n_queries
            else (self.dataset_root / "Audio" / audio_name)
        )
        audio_path = audio_path.with_suffix(".wav")
        wav, _ = apply_effects_file(
            str(audio_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
                ["vad", "-T", "0.25", "-p", "0.1"],
                ["reverse"],
                ["vad", "-T", "0.25", "-p", "0.1"],
                ["reverse"],
                ["pad", "0", "3"],
            ],
        )
        segments = wav.squeeze(0).unfold(0, 48000, 12000).unbind(0)
        return segments, len(segments), audio_name

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        segments, lengths, audio_names = zip(*samples)
        segments = [seg for segs in segments for seg in segs]
        return segments, (lengths, audio_names)


def parse_ecf(ecf_path):
    """Find audio paths from sws2013.ecf.xml."""

    root = ET.parse(str(ecf_path)).getroot()

    audio_names = []
    for excerpt in root.findall("excerpt"):
        audio_name = (
            excerpt.attrib["audio_filename"].replace("Audio/", "").replace(".wav", "")
        )
        audio_names.append(audio_name)

    return audio_names


def parse_tlist(tlist_path):
    """Find audio paths from sws2013_eval.tlist.xml."""

    root = ET.parse(str(tlist_path)).getroot()

    audio_names = []
    for term in root.findall("term"):
        audio_names.append(term.attrib["termid"])

    return audio_names
