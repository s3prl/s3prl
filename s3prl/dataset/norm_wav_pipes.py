from dataclasses import dataclass

from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class NormWavDecibel(DataPipe):
    target_level: int = -25
    wav_name: str = "wav"
    norm_wav_name: str = "wav"
    """
    Args:
        target_level (int): normalize the wav decibel level to the target value
        wav_name (str): handle for the `takes` (input)
        norm_wav_name (str): handle for the `provides` (output)
    """

    def normalize_wav_decibel(self, wav):
        wav = wav.squeeze()  # (seq_len, 1) -> (seq_len,)
        if self.target_level == 0:
            return wav
        rms = wav.pow(2).mean().pow(0.5)
        scalar = (10 ** (self.target_level / 20)) / (rms + 1e-10)
        wav = wav * scalar
        return wav.unsqueeze(1)  # (seq_len,) -> (seq_len, 1)

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        dataset.add_dynamic_item(
            self.normalize_wav_decibel, takes=self.wav_name, provides=self.norm_wav_name
        )
        return dataset
