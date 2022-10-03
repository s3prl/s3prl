"""
HEAR Competition submission script following the
https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api
guidelines
"""

from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram

from ..byol_a.augmentations import PrecomputedNorm
from ..byol_a.common import load_yaml_config
from ..byol_a.models.audio_ntt import AudioNTT2020
from ..byol_a.models.clstm import CLSTM
from ..byol_a.models.cvt import CvT
from ..byol_a.models.resnetish import resnetish34
from .utils import *

# Default frame duration in milliseconds
TIMESTAMP_FRAME_DUR = 1000
# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50

# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512


def get_model(model_name: str = "", cfg={}) -> torch.nn.Module:
    """Define the model object.

    Parameters
    ----------
    model_name: str, the name for pretrained model
    cfg: dict, the cfg parameters

    Returns
    -------
    torch.nn.Module object or a tensorflow "trackable" object
    """
    if model_name == "default":
        model = AudioNTT2020(n_mels=cfg.n_mels, d=cfg.feature_d)

    elif model_name == "resnetish34":
        model = resnetish34()

    elif model_name == "clstm":
        model = CLSTM()

    elif model_name == "cvt":
        s1_depth, s2_depth, s3_depth = cfg.depths
        s1_emb_dim, s2_emb_dim, s3_emb_dim = cfg.embed_dims
        s1_mlp_mult, s2_mlp_mult, s3_mlp_mult = cfg.mlp_mults

        model = CvT(
            s1_emb_dim=s1_emb_dim,
            s1_depth=s1_depth,
            s1_mlp_mult=s1_mlp_mult,
            s2_emb_dim=s2_emb_dim,
            s2_depth=s2_depth,
            s2_mlp_mult=s2_mlp_mult,
            s3_emb_dim=s3_emb_dim,
            s3_depth=s3_depth,
            s3_mlp_mult=s3_mlp_mult,
            pool=cfg.cvt_pool,
        )
    else:
        raise ValueError("Model not found.")
    return model


def load_model(
    model_file_path: str = "",
    model_name: str = "default",
    cfg_path: str = None,
) -> torch.nn.Module:
    """Load pre-trained DL models.

    Parameters
    ----------
    model_name: str, the name for pretrained model
    model_file_path: str, the path for pretrained model
    cfg_path: str, the path for yaml file including parameters value

    Returns
    -------
    torch.nn.Module object or a tensorflow "trackable" object
        Model loaded with pre-training weights
    """
    cfg_path = cfg_path or Path(__file__).parent / "config.yaml"
    # assert model_name in model_file_path.split('_')[0], "The checkpoint doesn't match with the selected model name"

    # Load config file
    cfg = load_yaml_config(cfg_path)

    # Load pretrained weights.
    model = get_model(model_name, cfg)

    state_dict = torch.load(model_file_path)
    model.load_state_dict(state_dict)
    return model


def get_timestamp_embeddings(
    audio_list: List,
    model: torch.nn.Module,
    frame_duration: float = TIMESTAMP_FRAME_DUR,
    hop_size: float = TIMESTAMP_HOP_SIZE,
    cfg_path: str = None,
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.
    Args:
        audio_list: List of torch tensor audios.
        model: Loaded model.
        frame_duration: Frame (segement) duration in milliseconds
        hop_size: Hop size in milliseconds.
            NOTE: Not required by the HEAR API. We add this optional parameter
            to improve the efficiency of scene embedding.
        cfg_path: str, the path for yaml file including parameters value
    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output. Shape: (n_sounds, n_timestamps).
    """
    cfg_path = cfg_path or Path(__file__).parent / "config.yaml"

    # Load config file
    cfg = load_yaml_config(cfg_path)
    to_melspec = MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    ).to(audio_list[0].device)

    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio_list[0].device)

    # Split the input audio signals into frames and then flatten to create a tensor
    # of audio frames that can be batch processed.
    frames, timestamps = frame_audio(
        audio_list,
        frame_size=(frame_duration / 1000) * cfg.sample_rate,
        hop_size=hop_size,
        sample_rate=cfg.sample_rate,
    )
    audio_batches, num_frames, _ = frames.shape
    frames = frames.flatten(end_dim=1)

    # Convert audio frames to Log Mel-spectrograms
    melspec_frames = (to_melspec(frames) + torch.finfo(torch.float).eps).log()
    normalizer = PrecomputedNorm(compute_timestamp_stats(melspec_frames))
    melspec_frames = normalizer(melspec_frames).unsqueeze(0)
    melspec_frames = melspec_frames.permute(1, 0, 2, 3)

    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(melspec_frames)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    # Disable parameter tuning
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    with torch.no_grad():
        embeddings_list = [model(batch[0]) for batch in loader]

    # Concatenate mini-batches back together and unflatten the frames
    # to reconstruct the audio batches
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = embeddings.unflatten(0, (audio_batches, num_frames))

    return embeddings, timestamps


def get_scene_embeddings(
    audio_list: List,
    model: torch.nn.Module,
    cfg_path: str = None,
) -> Tensor:
    """
    This function returns a single embedding for each audio clip. In this baseline
    implementation we simply summarize the temporal embeddings from
    get_timestamp_embeddings() using torch.mean().
    Args:
        audio_list: list of torch tensor audios (audios should be resampled to 16kHz).
        model: Loaded model.
        cfg_path:
    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    cfg_path = cfg_path or Path(__file__).parent / "config.yaml"

    device = audio_list[0].device
    # Load config file
    cfg = load_yaml_config(cfg_path)
    to_melspec = MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    ).to(device)
    stats = compute_scene_stats(audio_list, to_melspec)
    normalizer = PrecomputedNorm(stats)
    model = model.to(device)
    embeddings = generate_byols_embeddings(
        model, audio_list, to_melspec, normalizer, device
    )
    return embeddings
