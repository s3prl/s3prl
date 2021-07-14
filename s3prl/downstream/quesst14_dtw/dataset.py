import re
from pathlib import Path

from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file


class QUESST14Dataset(Dataset):
    """QUESST 2014 dataset (English-only)."""

    def __init__(self, split, **kwargs):
        dataset_root = Path(kwargs["dataset_root"])
        doc_paths = english_audio_paths(dataset_root, "language_key_utterances.lst")
        query_paths = english_audio_paths(dataset_root, f"language_key_{split}.lst")

        self.dataset_root = dataset_root
        self.n_queries = len(query_paths)
        self.n_docs = len(doc_paths)
        self.data = query_paths + doc_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]
        wav, _ = apply_effects_file(
            str(audio_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["gain", "-3.0"],
            ],
        )
        wav = wav.squeeze(0)
        return wav.numpy(), audio_path.with_suffix("").name

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        wavs, audio_names = zip(*samples)
        return wavs, audio_names


def english_audio_paths(dataset_root_path, lst_name):
    """Extract English audio paths."""
    audio_paths = []

    with open(dataset_root_path / "scoring" / lst_name) as f:
        for line in f:
            audio_path, lang = tuple(line.strip().split())
            if lang != "nnenglish":
                continue
            audio_path = re.sub(r"^.*?\/", "", audio_path)
            audio_paths.append(dataset_root_path / audio_path)

    return audio_paths
