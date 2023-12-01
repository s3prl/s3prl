import random
import logging
from typing import List, Tuple

import torch
import torchaudio
import torch.fft as fft
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


def normalize(waveforms, lengths=None, amp_type="avg", eps=1e-14):
    """This function normalizes a signal to unitary average or peak amplitude.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether one wants to normalize with respect to "avg" or "peak"
        amplitude. Choose between ["avg", "peak"]. Note: for "avg" clipping
        is not prevented and can occur.
    eps : float
        A small number to add to the denominator to prevent NaN.

    Returns
    -------
    waveforms : tensor
        Normalized level waveform.
    """

    assert amp_type in ["avg", "peak"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    den = compute_amplitude(waveforms, lengths, amp_type) + eps
    if batch_added:
        waveforms = waveforms.squeeze(0)
    return waveforms / den


def rescale(waveforms, lengths, target_lvl, amp_type="avg", scale="linear"):
    """This functions performs signal rescaling to a target level.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    target_lvl : float
        Target lvl in dB or linear scale.
    amp_type : str
        Whether one wants to rescale with respect to "avg" or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        whether target_lvl belongs to linear or dB scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    waveforms : tensor
        Rescaled waveforms.
    """

    assert amp_type in ["peak", "avg"]
    assert scale in ["linear", "dB"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    waveforms = normalize(waveforms, lengths, amp_type)

    if scale == "linear":
        out = target_lvl * waveforms
    elif scale == "dB":
        out = dB_to_amplitude(target_lvl) * waveforms

    else:
        raise NotImplementedError("Invalid scale, choose between dB and linear")

    if batch_added:
        out = out.squeeze(0)

    return out


def convolve1d(
    waveform,
    kernel,
    padding=0,
    pad_type="constant",
    stride=1,
    groups=1,
    use_fft=False,
    rotation_index=0,
):
    """Use torch.nn.functional to perform 1d padding and conv.

    Arguments
    ---------
    waveform : tensor
        The tensor to perform operations on.
    kernel : tensor
        The filter to apply during convolution.
    padding : int or tuple
        The padding (pad_left, pad_right) to apply.
        If an integer is passed instead, this is passed
        to the conv1d function and pad_type is ignored.
    pad_type : str
        The type of padding to use. Passed directly to
        `torch.nn.functional.pad`, see PyTorch documentation
        for available options.
    stride : int
        The number of units to move each time convolution is applied.
        Passed to conv1d. Has no effect if `use_fft` is True.
    groups : int
        This option is passed to `conv1d` to split the input into groups for
        convolution. Input channels should be divisible by the number of groups.
    use_fft : bool
        When `use_fft` is passed `True`, then compute the convolution in the
        spectral domain using complex multiply. This is more efficient on CPU
        when the size of the kernel is large (e.g. reverberation). WARNING:
        Without padding, circular convolution occurs. This makes little
        difference in the case of reverberation, but may make more difference
        with different kernels.
    rotation_index : int
        This option only applies if `use_fft` is true. If so, the kernel is
        rolled by this amount before convolution to shift the output location.

    Returns
    -------
    The convolved waveform.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0).unsqueeze(2)
    >>> kernel = torch.rand(1, 10, 1)
    >>> signal = convolve1d(signal, kernel, padding=(9, 0))
    """
    if len(waveform.shape) != 3:
        raise ValueError("Convolve1D expects a 3-dimensional tensor")

    # Move time dimension last, which pad and fft and conv expect.
    waveform = waveform.transpose(2, 1)
    kernel = kernel.transpose(2, 1)

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        waveform = torch.nn.functional.pad(input=waveform, pad=padding, mode=pad_type)

    # This approach uses FFT, which is more efficient if the kernel is large
    if use_fft:
        # Pad kernel to same length as signal, ensuring correct alignment
        zero_length = waveform.size(-1) - kernel.size(-1)

        # Handle case where signal is shorter
        if zero_length < 0:
            kernel = kernel[..., :zero_length]
            zero_length = 0

        # Perform rotation to ensure alignment
        zeros = torch.zeros(
            kernel.size(0), kernel.size(1), zero_length, device=kernel.device
        )
        after_index = kernel[..., rotation_index:]
        before_index = kernel[..., :rotation_index]
        kernel = torch.cat((after_index, zeros, before_index), dim=-1)

        result = fft.rfft(waveform) * fft.rfft(kernel)
        convolved = fft.irfft(result, n=waveform.size(-1))

    # Use the implementation given by torch, which should be efficient on GPU
    else:
        convolved = torch.nn.functional.conv1d(
            input=waveform,
            weight=kernel,
            stride=stride,
            groups=groups,
            padding=padding if not isinstance(padding, tuple) else 0,
        )

    # Return time dimension to the second dimension.
    return convolved.transpose(2, 1)


def reverberate(waveforms, rir_waveform, rescale_amp="avg"):
    """
    General function to contaminate a given signal with reverberation given a
    Room Impulse Response (RIR).
    It performs convolution between RIR and signal, but without changing
    the original amplitude of the signal.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    rir_waveform : tensor
        RIR tensor, shape should be [batch, time].
    rescale_amp : str
        Whether reverberated signal is rescaled (None) and with respect either
        to original signal "peak" amplitude or "avg" average amplitude.
        Choose between [None, "avg", "peak"].

    Returns
    -------
    waveforms: tensor
        Reverberated signal.

    """

    orig_shape = waveforms.shape

    if len(waveforms.shape) > 3 or len(rir_waveform.shape) > 3:
        raise NotImplementedError

    # if inputs are mono tensors we reshape to 1, samples
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0).unsqueeze(-1)
    elif len(waveforms.shape) == 2:
        waveforms = waveforms.unsqueeze(-1)

    if len(rir_waveform.shape) == 1:  # convolve1d expects a 3d tensor !
        rir_waveform = rir_waveform.unsqueeze(0).unsqueeze(-1)
    elif len(rir_waveform.shape) == 2:
        rir_waveform = rir_waveform.unsqueeze(-1)

    # Compute the average amplitude of the clean
    orig_amplitude = compute_amplitude(waveforms, waveforms.size(1), rescale_amp)

    # Compute index of the direct signal, so we can preserve alignment
    value_max, direct_index = rir_waveform.abs().max(axis=1, keepdim=True)

    # Making sure the max is always positive (if not, flip)
    # mask = torch.logical_and(rir_waveform == value_max,  rir_waveform < 0)
    # rir_waveform[mask] = -rir_waveform[mask]

    # Use FFT to compute convolution, because of long reverberation filter
    waveforms = convolve1d(
        waveform=waveforms,
        kernel=rir_waveform,
        use_fft=True,
        rotation_index=direct_index,
    )

    # Rescale to the peak amplitude of the clean waveform
    waveforms = rescale(waveforms, waveforms.size(1), orig_amplitude, rescale_amp)

    if len(orig_shape) == 1:
        waveforms = waveforms.squeeze(0).squeeze(-1)
    if len(orig_shape) == 2:
        waveforms = waveforms.squeeze(-1)

    return waveforms


class DistortedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        batch_sampler,
        collate_fn,
        noise_paths: List[str] = None,
        snrs: List[float] = None,
        reverb_paths: List[str] = None,
        seed: int = 0,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.noise_paths = noise_paths
        self.reverb_paths = reverb_paths
        self.snrs = snrs
        self.sample_rate = sample_rate

        self.batch_indices = list(self.batch_sampler)

        seed_randomizer = random.Random(seed)
        self.seeds = list(range(len(self.batch_indices)))
        seed_randomizer.shuffle(self.seeds)

        self.add_noise = (noise_paths is not None) and (snrs is not None)
        self.add_reverb = reverb_paths is not None

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

            if self.add_reverb:
                reverb_path = reverb_randomizer.choice(self.reverb_paths)
                reverb, reverb_sr = torchaudio.load(reverb_path)

                if reverb_sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        reverb_sr, self.sample_rate
                    )
                    reverb = resampler(reverb)

                distorted_wav = reverberate(
                    distorted_wav.reshape(1, -1, 1), reverb.reshape(1, -1)
                )
                distorted_wav = distorted_wav.view(-1)

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
    reverb_paths = None

    noise_conf = distortion_conf.get("noise")
    if noise_conf is not None:
        logger.info(f"Addings noises to the dataloader")
        noise_paths = read_lines_to_list(noise_conf["audios"])
        snrs = noise_conf["snrs"]

    reverb_conf = distortion_conf.get("reverb")
    if reverb_conf is not None:
        logger.info(f"Adding reverberation to the dataloader")
        reverb_paths = read_lines_to_list(reverb_conf["rirs"])

    distorted_dataset = DistortedDataset(
        dataset, batch_sampler, collate_fn, noise_paths, snrs, reverb_paths, seed
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
