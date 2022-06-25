import random
import torchaudio
from typing import List
from dataclasses import dataclass
from speechbrain.processing.signal_processing import compute_amplitude, dB_to_amplitude

import torch

from s3prl.base.container import Container

from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class AdditiveNoise(DataPipe):
    noise_paths: List[str] = None
    snrs: List[float] = None
    sample_rate: int = 16000
    repeat: bool = True
    noise_paths_hook: dict = None

    wav_name: str = "wav"
    noisy_name: str = "wav_noisy"

    def add_noise(self, target, noise_paths):
        snrs = self.snrs or [-3, 0, 3]
        noise_path = random.sample(noise_paths, k=1)[0]

        assert target.dim() == 2 and target.size(-1) == 1
        noise, sr = torchaudio.load(noise_path)
        if sr != self.sample_rate:
            noise = torchaudio.transforms.Resample(sr, self.sample_rate)(noise)
        noise = noise.view(-1, 1)

        target, target_len = target.view(1, -1, 1), target.new_ones(1) * target.size(0)
        noise, noise_len = noise.view(1, -1, 1), noise.new_ones(1) * noise.size(0)

        clean_amplitude = compute_amplitude(target, target_len)

        SNR = torch.ones(len(target)) * random.sample(snrs, k=1)[0]
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        target *= 1 - noise_amplitude_factor

        if target.size(1) > noise.size(1) and not self.repeat:
            start = random.randint(0, target.size(1) - noise.size(1))
            pre_pad = noise.new_zeros(1, start, 1)
            post_pad = noise.new_zeros(1, target.size(1) - (start + noise.size(1)), 1)
            noise = torch.cat((pre_pad, noise, post_pad), dim=1)
        else:
            if target.size(1) > noise.size(1):
                num_repeat = target.size(1) // noise.size(1) + 1
                noise = noise.expand(num_repeat, -1, -1).reshape(1, -1, 1)

            start = random.randint(0, noise.size(1) - target.size(1))
            noise = noise[:, start : start + target.size(1), :]
        assert noise.size(1) == target_len.item()

        noise_amplitude = compute_amplitude(noise, target_len)
        noise *= new_noise_amplitude / (noise_amplitude + 1e-14)

        noisy_waveform = target + noise
        return noisy_waveform.view(-1, 1)

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        noise_paths = self.noise_paths or Container(self.noise_paths_hook).instantiate()
        dataset.add_tool("noise_paths", noise_paths)
        dataset.add_dynamic_item(
            self.add_noise,
            takes=[self.wav_name, "noise_paths"],
            provides=self.noisy_name,
        )
        dataset.replace_output_key(self.wav_name, self.noisy_name)
        return dataset


@dataclass
class ApplySoxEffects(DataPipe):
    effects: list = None
    sample_rate: int = 16000

    wav_name: str = "wav"
    soxed_name: str = "wav_soxed"

    def apply_sox_effects(self, wav):
        wav, sr = torchaudio.sox_effects.apply_effects_tensor(
            wav.view(1, -1),
            self.sample_rate,
            effects=self.effects,
        )
        return wav.view(-1, 1)

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        dataset.add_dynamic_item(
            self.apply_sox_effects,
            takes=self.wav_name,
            provides=self.soxed_name,
        )
        dataset.replace_output_key(self.wav_name, self.soxed_name)
        return dataset


class Reverberation(ApplySoxEffects):
    def __init__(
        self,
        reverberance: float = 50,
        hf_damping: float = 50,
        room_scale: float = 100,
        stereo_depth: float = 100,
        sample_rate: int = 16000,
        wav_name: str = "wav",
        reverbed_name: str = "wav_reverb",
    ):
        super().__init__(
            effects=[
                ["gain", "-3"],
                [
                    "reverb",
                    str(reverberance),
                    str(hf_damping),
                    str(room_scale),
                    str(stereo_depth),
                ],
                ["channels", "1"],
            ],
            sample_rate=sample_rate,
            wav_name=wav_name,
            soxed_name=reverbed_name,
        )


class ShiftPitchAndResample(ApplySoxEffects):
    def __init__(
        self,
        shift_cent: int = 0,
        quick: bool = False,
        sample_rate: int = 16000,
        wav_name: str = "wav",
        pitched_name: str = "wav_pitch",
    ):
        command = ["pitch", str(shift_cent)]
        if quick:
            command.insert(1, "-q")
        super().__init__(
            effects=[command, ["channels", "1"], ["rate", "16000"]],
            sample_rate=sample_rate,
            wav_name=wav_name,
            soxed_name=pitched_name,
        )
