import random
import logging
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def read_lines_to_list(filepath: str):
    with open(filepath) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def dB_to_amplitude(SNR):
    """
    Copied from SpeechBrain

    Returns the amplitude ratio, converted from decibels.

    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.

    Example
    -------
    >>> round(dB_to_amplitude(SNR=10), 3)
    3.162
    >>> dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 20)


def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    """
    Copied from SpeechBrain

    Compute amplitude of a batch of waveforms.

    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
        Shape should be `[time]` or `[batch, time]` or
        `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether to compute "avg" average or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        Whether to compute amplitude in "dB" or "linear" scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    The average amplitude of the waveforms.

    Example
    -------
    >>> signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    >>> compute_amplitude(signal, signal.size(1))
    tensor([[0.6366]])
    """
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0)

    assert amp_type in ["avg", "peak"]
    assert scale in ["linear", "dB"]

    if amp_type == "avg":
        if lengths is None:
            out = torch.mean(torch.abs(waveforms), dim=1, keepdim=True)
        else:
            wav_sum = torch.sum(input=torch.abs(waveforms), dim=1, keepdim=True)
            out = wav_sum / lengths
    elif amp_type == "peak":
        out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)[0]
    else:
        raise NotImplementedError

    if scale == "linear":
        return out
    elif scale == "dB":
        return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
    else:
        raise NotImplementedError


def augment_noise(
    wav: torch.Tensor,
    noise: torch.Tensor,
    snr: int,
    randomizer: random.Random,
    normalize: bool = True,
):
    """Modified from SpeechBrain"""

    if len(wav) > len(noise):
        mutiplier = len(wav) // len(noise) + 1
        noise = noise.view(1, -1).repeat(mutiplier, 1).view(-1)

    start = randomizer.choice(list(range(0, len(noise) - len(wav))))
    noise_waveform = noise[start : start + len(wav)].view(1, -1, 1)

    # Copy clean waveform to initialize noisy waveform
    noisy_waveform = wav.clone().view(1, -1, 1)

    # Compute the average amplitude of the clean waveforms
    clean_amplitude = compute_amplitude(noisy_waveform)

    noise_amplitude_factor = 1 / (dB_to_amplitude(snr) + 1)
    new_noise_amplitude = noise_amplitude_factor * clean_amplitude

    # Scale clean signal appropriately
    noisy_waveform *= 1 - noise_amplitude_factor

    # Rescale and add
    noise_amplitude = compute_amplitude(noise_waveform)
    noise_waveform *= new_noise_amplitude / (noise_amplitude + 1e-14)
    noisy_waveform += noise_waveform

    # Normalizing to prevent clipping
    if normalize:
        abs_max, _ = torch.max(torch.abs(noisy_waveform), dim=1, keepdim=True)
        noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

    return noisy_waveform.view(-1)


class DistortedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        batch_sampler,
        collate_fn,
        noise_paths: List[str] = None,
        snrs: List[float] = None,
        reverberance: Tuple[float] = None,
        seed: int = 0,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.noise_paths = noise_paths
        self.reverberance = reverberance
        self.snrs = snrs
        self.sample_rate = sample_rate

        self.batch_indices = list(self.batch_sampler)

        seed_randomizer = random.Random(seed)
        self.seeds = list(range(len(self.batch_indices)))
        seed_randomizer.shuffle(self.seeds)

        self.add_noise = (noise_paths is not None) and (snrs is not None)
        self.add_reverb = reverberance is not None

    def __len__(self):
        return len(self.batch_indices)

    def __getitem__(self, index: int):
        torchaudio.set_audio_backend("sox_io")
        noise_randomizer = random.Random(self.seeds[index])
        reverb_randomizer = random.Random(self.seeds[index])

        indices = self.batch_indices[index]
        data_points = [self.dataset[indice] for indice in indices]
        all_wavs, *others = self.collate_fn(data_points)

        distorted_wavs = []
        for wav in all_wavs:
            distorted_wav = torch.FloatTensor(wav).clone()

            if self.add_noise:
                noise_path = noise_randomizer.choice(self.noise_paths)
                noise, noise_sr = torchaudio.load(noise_path)

                if noise_sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        noise_sr, self.sample_rate
                    )
                    noise = resampler(noise)

                noise = noise.view(-1)

                snr = noise_randomizer.choice(self.snrs)
                distorted_wav = augment_noise(
                    distorted_wav, noise, snr, noise_randomizer
                )

            if self.add_reverb:
                reverberance = reverb_randomizer.uniform(*self.reverberance)
                distorted_wav, sr = torchaudio.sox_effects.apply_effects_tensor(
                    distorted_wav.view(1, -1),
                    self.sample_rate,
                    [["reverb", f"{reverberance}"]],
                )
                distorted_wav = distorted_wav.view(-1)

            distorted_wavs.append(distorted_wav.numpy())

        return [distorted_wavs, *others]


def make_distorted_dataloader(
    dataloader: DataLoader, distortion_conf: dict, seed: int = 0
):
    dataset = dataloader.dataset
    num_workers = dataloader.num_workers
    prefetch_factor = dataloader.prefetch_factor
    pin_memory = dataloader.pin_memory
    timeout = dataloader.timeout
    worker_init_fn = dataloader.worker_init_fn
    multiprocessing_context = dataloader.multiprocessing_context
    batch_sampler = dataloader.batch_sampler
    collate_fn = dataloader.collate_fn
    generator = dataloader.generator
    persistent_workers = dataloader.persistent_workers

    noise_paths = None
    snrs = None

    noise_conf = distortion_conf.get("noise")
    if noise_conf is not None:
        logger.info(f"Addings noises to the dataloader")
        noise_paths = read_lines_to_list(noise_conf["audios"])
        snrs = noise_conf["snrs"]

    reverb_conf = distortion_conf.get("reverb")
    if reverb_conf is not None:
        logger.info(f"Adding reverberation to the dataloader")
        reverberance = reverb_conf["reverberance"]

    distorted_dataset = DistortedDataset(
        dataset, batch_sampler, collate_fn, noise_paths, snrs, reverberance, seed
    )
    distorted_dataloader = DataLoader(
        distorted_dataset,
        batch_size=1,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        collate_fn=lambda xs: xs[0],
    )
    return distorted_dataloader
