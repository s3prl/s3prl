import pytest
import torchaudio
import pandas as pd
from pathlib import Path
from librosa.util import find_files
from s3prl.dataset.base import SequentialDataPipe
from s3prl.dataset.common_pipes import LoadAudio
from s3prl.dataset.effects import AdditiveNoise, Reverberation, ShiftPitchAndResample


@pytest.mark.practice
def test_add_noise():
    noises = find_files("/home/leo/d/datasets/noise_data/NOISEX-92/")
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
            snrs=[-9, -6, -3],
            wav_name="wav_pitch",
        ),
        Reverberation(reverberance=100, hf_damping=100, wav_name="wav_noisy"),
    )
    dataset = pipes(data)
    noisy = dataset[0]["wav_pitch"]
    torchaudio.save("pitch.wav", noisy.view(1, -1), sample_rate=16000)
