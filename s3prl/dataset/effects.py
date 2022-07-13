import random
from dataclasses import dataclass
from typing import List

import torch
import torchaudio
from speechbrain.processing.signal_processing import compute_amplitude, dB_to_amplitude

from s3prl.base.container import Container

from .base import AugmentedDynamicItemDataset, DataPipe

NUM_ALL_SEED = 1000000


class AdditiveNoise(DataPipe):
    def __init__(
        self,
        noise_paths: List[str] = None,
        snrs: tuple = (3, 9),
        sample_rate: int = 16000,
        repeat: bool = True,
        noise_paths_hook: dict = None,
        wav_name: str = "wav",
        noisy_name: str = "wav_noisy",
        seed: int = 0,
        **kwds,
    ):
        self.noise_paths = noise_paths
        self.snrs = snrs
        self.sample_rate = sample_rate
        self.repeat = repeat
        self.noise_paths_hook = noise_paths_hook
        self.wav_name = wav_name
        self.noisy_name = noisy_name
        self.seed = seed

    def add_noise(self, target, data_id: str, seeds: List[int]):
        assert target.dim() == 2 and target.size(-1) == 1
        seed = seeds[data_id]
        random.seed(seed)

        noise_paths = self.noise_paths or Container(self.noise_paths_hook)()
        noise_path = random.sample(sorted(noise_paths), k=1)[0]
        snr = random.uniform(self.snrs[0], self.snrs[1])
        repeat = self.repeat
        sample_rate = self.sample_rate

        noise, sr = torchaudio.load(noise_path)
        if sr != sample_rate:
            noise = torchaudio.transforms.Resample(sr, sample_rate)(noise)
        noise = noise.view(-1, 1)

        target, target_len = target.view(1, -1, 1), target.new_ones(1) * target.size(0)
        noise, noise_len = noise.view(1, -1, 1), noise.new_ones(1) * noise.size(0)

        clean_amplitude = compute_amplitude(target, target_len)

        noise_amplitude_factor = 1 / (dB_to_amplitude(snr) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        target *= 1 - noise_amplitude_factor

        if target.size(1) > noise.size(1) and not repeat:
            start = random.randint(0, target.size(1) - noise.size(1))
            pre_pad = noise.new_zeros(1, start, 1)
            post_pad = noise.new_zeros(1, target.size(1) - (start + noise.size(1)), 1)
            noise = torch.cat((pre_pad, noise, post_pad), dim=1)
        else:
            if target.size(1) > noise.size(1):
                num_repeat = target.size(1) // noise.size(1) + 1
                noise = noise.expand(num_repeat, -1, -1).reshape(1, -1, 1)

            start = random.randint(0, target.size(1) - noise.size(1))
            noise = noise[:, start : start + target.size(1), :]
        assert noise.size(1) == target_len.item()

        noise_amplitude = compute_amplitude(noise, target_len)
        noise *= new_noise_amplitude / (noise_amplitude + 1e-14)

        noisy_waveform = target + noise
        return noisy_waveform.view(-1, 1)

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        seeds = dict()
        all_seeds = list(range(NUM_ALL_SEED))
        random.seed(self.seed)
        random.shuffle(all_seeds)
        with dataset.output_keys_as(["id"]):
            for data_index, item in enumerate(dataset):
                seeds[item["id"]] = all_seeds[data_index]

        dataset.add_tool("seeds", seeds)
        dataset.add_dynamic_item(
            self.add_noise,
            takes=[self.wav_name, "id", "seeds"],
            provides=self.noisy_name,
        )
        dataset.replace_output_key(self.wav_name, self.noisy_name)
        return dataset


@dataclass
class ApplySoxEffects(DataPipe):
    effects: List[List[str]] = None
    sample_rate: int = 16000

    wav_name: str = "wav"
    soxed_name: str = "wav_soxed"

    def apply_sox_effects(self, wav, data_id, id2effects: dict):
        wav, sr = torchaudio.sox_effects.apply_effects_tensor(
            wav.view(1, -1),
            self.sample_rate,
            effects=id2effects[data_id],
        )
        return wav.view(-1, 1)

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        if len(self.effects) < len(dataset):
            effects = self.effects * (len(dataset) // len(self.effects) + 1)
        else:
            effects = self.effects
        effects = effects[: len(dataset)]
        id2effects = dict()
        with dataset.output_keys_as(["id"]):
            for data_index, item in enumerate(dataset):
                id2effects[item["id"]] = effects[data_index]
        dataset.add_tool("id2effects", id2effects)
        dataset.add_dynamic_item(
            self.apply_sox_effects,
            takes=[self.wav_name, "id", "id2effects"],
            provides=self.soxed_name,
        )
        dataset.replace_output_key(self.wav_name, self.soxed_name)
        return dataset


class Reverberation(ApplySoxEffects):
    def __init__(
        self,
        reverberance: tuple = (50, 100),
        hf_damping: tuple = (50, 100),
        room_scale: float = (100, 100),
        stereo_depth: float = (100, 100),
        sample_rate: int = 16000,
        wav_name: str = "wav",
        reverbed_name: str = "wav_reverb",
        seed: int = 0,
        **kwds,
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
        self.reverberance = reverberance
        self.hf_damping = hf_damping
        self.room_scale = room_scale
        self.stereo_depth = stereo_depth
        self.seed = seed

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        effects = []
        random.seed(self.seed)
        with dataset.output_keys_as(["id"]):
            for _ in dataset:
                effects.append(
                    [
                        ["gain", "-3"],
                        [
                            "reverb",
                            str(
                                random.uniform(
                                    self.reverberance[0], self.reverberance[1]
                                )
                            ),
                            str(random.uniform(self.hf_damping[0], self.hf_damping[1])),
                            str(random.uniform(self.room_scale[0], self.room_scale[1])),
                            str(
                                random.uniform(
                                    self.stereo_depth[0], self.stereo_depth[1]
                                )
                            ),
                        ],
                        ["channels", "1"],
                    ]
                )
        self.effects = effects
        return super().forward(dataset)


class ShiftPitchAndResample(ApplySoxEffects):
    def __init__(
        self,
        shift_cent: int = 0,
        quick: bool = False,
        sample_rate: int = 16000,
        wav_name: str = "wav",
        pitched_name: str = "wav_pitch",
        **kwds,
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
