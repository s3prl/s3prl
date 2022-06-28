from pathlib import Path

import pandas as pd
import pytest
import torchaudio
from librosa.util import find_files

from s3prl.dataset.base import SequentialDataPipe
from s3prl.dataset.common_pipes import LoadAudio
from s3prl.dataset.effects import AdditiveNoise, Reverberation, ShiftPitchAndResample


@pytest.mark.practice
def test_add_noise():
    data = {
        "1": dict(
            wav_path="/home/leo/d/datasets/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
        ),
    }
    dataset = AdditiveNoise(
        noise_paths_hook=dict(
            _cls=find_files, directory="/home/leo/d/datasets/noise_data/NOISEX-92/"
        ),
        snrs=[0],
    )(LoadAudio(audio_sample_rate=16000)(data))
    noisy = dataset[0]["wav_noisy"]
    torchaudio.save("noisy.wav", noisy.view(1, -1), sample_rate=16000)


@pytest.mark.practice
def test_noise_and_reverb_and_pitch():
    data = {
        "1": dict(
            wav_path="/home/leo/d/datasets/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
        ),
    }
    pipes = SequentialDataPipe(
        LoadAudio(audio_sample_rate=16000),
        ShiftPitchAndResample(shift_cent=600, sample_rate=16000, wav_name="wav"),
        AdditiveNoise(
            noise_paths_hook=dict(
                _cls="librosa.util.find_files",
                directory="/home/leo/d/datasets/noise_data/NOISEX-92/",
            ),
            snrs=(-3, 3),
            wav_name="wav_pitch",
            seed=113,
            repeat=False,
        ),
        Reverberation(reverberance=(70, 80), hf_damping=(80, 90), wav_name="wav_noisy"),
    )
    dataset = pipes(data)
    for data_id, data in enumerate(dataset):
        reverb = data["wav_reverb"]
        torchaudio.save(f"{data_id}_reverb.wav", reverb.view(1, -1), sample_rate=16000)
