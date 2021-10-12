from random import randint
from pathlib import Path

from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file

CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "_unknown_",
    "_silence_",
]

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SpeechCommandsBaseDataset(Dataset):
    """12-class Speech Commands base dataset."""

    def __init__(self):
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = 12
        self.data = []

    def __getitem__(self, idx):
        class_name, audio_path = self.data[idx]
        wav, _ = apply_effects_file(str(audio_path), EFFECTS)
        wav = wav.squeeze(0).numpy()
        fileid = "-".join(Path(audio_path).parts[-2:])
        return wav, self.class2index[class_name], fileid

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        return zip(*samples)


class SpeechCommandsDataset(SpeechCommandsBaseDataset):
    """Training and validation dataset."""

    def __init__(self, data_list, **kwargs):
        super().__init__()

        data = [
            (class_name, audio_path)
            if class_name in self.class2index.keys()
            else ("_unknown_", audio_path)
            for class_name, audio_path in data_list
        ]
        data += [
            ("_silence_", audio_path)
            for audio_path in Path(
                kwargs["speech_commands_root"], "_background_noise_"
            ).glob("*.wav")
        ]

        class_counts = {class_name: 0 for class_name in CLASSES}
        for class_name, _ in data:
            class_counts[class_name] += 1

        sample_weights = [
            len(data) / class_counts[class_name] for class_name, _ in data
        ]

        self.data = data
        self.sample_weights = sample_weights

    def __getitem__(self, idx):
        wav, label, stem = super().__getitem__(idx)

        # _silence_ audios are longer than 1 sec.
        if label == self.class2index["_silence_"]:
            random_offset = randint(0, len(wav) - 16000)
            wav = wav[random_offset : random_offset + 16000]

        return wav, label, stem


class SpeechCommandsTestingDataset(SpeechCommandsBaseDataset):
    """Testing dataset."""

    def __init__(self, **kwargs):
        super().__init__()

        self.data = [
            (class_dir.name, audio_path)
            for class_dir in Path(kwargs["speech_commands_test_root"]).iterdir()
            if class_dir.is_dir()
            for audio_path in class_dir.glob("*.wav")
        ]
