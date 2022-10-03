"""
Utility functions for hear-kit
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


def compute_scene_stats(audios, to_melspec):
    mean = 0.0
    std = 0.0
    for audio in audios:
        # Compute log-mel-spectrogram
        lms = (to_melspec(audio) + torch.finfo(torch.float).eps).log()

        # Compute mean, std
        mean += lms.mean()
        std += lms.std()

    mean /= len(audios)
    std /= len(audios)
    stats = [mean.item(), std.item()]
    return stats


def compute_timestamp_stats(melspec):
    """Compute statistics of the mel-spectrograms.

    Parameters
    ----------
    melspec : Tensor of shape (n_sounds*n_frames, n_mels, time)

    Returns
    -------
    list containing the mean and the standard deviation of the mel-spectrograms"""
    # Compute mean, std
    mean = melspec.mean()
    std = melspec.std()
    mean /= len(melspec)
    std /= len(melspec)

    stats = [mean.item(), std.item()]
    return stats


def frame_audio(
    audio: Tensor, frame_size: int, hop_size: float, sample_rate: int
) -> Tuple[Tensor, Tensor]:
    """
    Adapted from https://github.com/neuralaudio/hear-baseline/hearbaseline/

    Slices input audio into frames that are centered and occur every
    sample_rate * hop_size samples. We round to the nearest sample.
    Args:
        audio: input audio, expects a 2d Tensor of shape:
            (n_sounds, num_samples)
        frame_size: the number of samples each resulting frame should be
        hop_size: hop size between frames, in milliseconds
        sample_rate: sampling rate of the input audio
    Returns:
        - A Tensor of shape (n_sounds, num_frames, frame_size)
        - A Tensor of timestamps corresponding to the frame centers with shape:
            (n_sounds, num_frames).
    """

    # Zero pad the beginning and the end of the incoming audio with half a frame number
    # of samples. This centers the audio in the middle of each frame with respect to
    # the timestamps.
    frame_size = int(frame_size)
    audio = F.pad(audio, (frame_size // 2, frame_size - frame_size // 2))
    num_padded_samples = audio.shape[1]

    frame_step = int(hop_size / 1000.0 * sample_rate)
    frame_number = 0
    frames = []
    timestamps = []
    frame_start = 0
    frame_end = frame_size
    while True:
        frames.append(audio[:, frame_start:frame_end])
        timestamps.append(frame_number * frame_step / sample_rate * 1000.0)

        # Increment the frame_number and break the loop if the next frame end
        # will extend past the end of the padded audio samples
        frame_number += 1
        frame_start = int(round(frame_number * frame_step))
        frame_end = frame_start + frame_size

        if not frame_end <= num_padded_samples:
            break

    # Expand out the timestamps to have shape (n_sounds, num_frames)
    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
    timestamps_tensor = timestamps_tensor.expand(audio.shape[0], -1)

    return torch.stack(frames, dim=1), timestamps_tensor


def generate_byols_embeddings(model, audios, to_melspec, normalizer, device):
    """
    Generate audio embeddings from a pretrained feature extractor.

    Converts audios to float, resamples them to the desired learning_rate,
    and produces the embeddings from a pre-trained model.

    Adapted from https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark

    Parameters
    ----------
    model : torch.nn.Module object or a tensorflow "trackable" object
        Model loaded with pre-training weights
    audios : list
        List of audios, loaded as a numpy arrays
    to_melspec : torchaudio.transforms.MelSpectrogram object
        Mel-spectrogram transform to create a spectrogram from an audio signal
    normalizer : nn.Module
        Pre-normalization transform
    device : torch.device object
        Used device (CPU or GPU)

    Returns
    ----------
    embeddings: Tensor
        2D Array of embeddings for each audio of size (N, M). N = number of samples, M = embedding dimension
    """
    embeddings = []
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    with torch.no_grad():
        for audio in tqdm(audios, desc=f"Generating Embeddings...", total=len(audios)):
            lms = normalizer(
                (
                    to_melspec(audio.to(device).unsqueeze(0))
                    + torch.finfo(torch.float).eps
                ).log()
            ).unsqueeze(0)
            embedding = model(lms)
            embeddings.append(embedding)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings
