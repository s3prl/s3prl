import re
from pathlib import Path

from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file


class QUESST14Dataset(Dataset):
    """QUESST 2014 dataset (English-only)."""

    def __init__(self, split, **kwargs):
        assert split in ["dev", "eval"]

        dataset_root = Path(kwargs["quesst2014_root"])
        doc_paths = get_audio_paths(dataset_root, "language_key_utterances.lst")
        query_paths = get_audio_paths(dataset_root, f"language_key_{split}.lst")

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
                ["norm"],
                ["vad", "-T", "0.25", "-p", "0.1"],
                ["reverse"],
                ["vad", "-T", "0.25", "-p", "0.1"],
                ["reverse"],
                ["pad", "0", "3"],
            ],
        )
        segments = wav.squeeze(0).unfold(0, 48000, 12000).unbind(0)
        return segments, len(segments), audio_path.with_suffix("").name

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        segments, lengths, audio_names = zip(*samples)
        segments = [seg for segs in segments for seg in segs]
        return segments, (lengths, audio_names)


def get_audio_paths(dataset_root_path, lst_name):
    """Extract audio paths."""
    audio_paths = []

    with open(dataset_root_path / "scoring" / lst_name) as f:
        for line in f:
            audio_path, lang = tuple(line.strip().split())
            # if lang != "nnenglish":
            #     continue
            audio_path = re.sub(r"^.*?\/", "", audio_path)
            audio_paths.append(dataset_root_path / audio_path)

    return audio_paths
