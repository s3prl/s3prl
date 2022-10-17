import logging
import os
from pathlib import Path

import torchaudio
from joblib import Parallel, delayed
from librosa.util import find_files
from tqdm import tqdm

logger = logging.getLogger(__name__)


def resample_hear_corpus(task_dir: str, target_sr: int = 16000, num_workers: int = 6):
    """
    Resample audio files in

    ${task_dir}/48000/

    to

    ${task_dir}/${target_sr}/
    """

    task_dir: Path = Path(task_dir)
    target_audio_dir: Path = task_dir / f"{target_sr}"
    if target_audio_dir.is_dir():
        logger.info(f"{target_audio_dir} already exist. Do not need to resample")
        return

    default_audio_dir = task_dir / "48000"
    assert default_audio_dir.exists(), f"{default_audio_dir} not found"

    split_names = os.listdir(default_audio_dir)
    for split_name in sorted(split_names):
        split_dir = default_audio_dir / split_name
        wav_paths = find_files(split_dir)
        tgt_dir = target_audio_dir / split_name
        tgt_dir.mkdir(exist_ok=True, parents=True)

        def resample(wav_path: str):
            wav, sr = torchaudio.load(wav_path)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                wav = resampler(wav)
            torchaudio.save(
                str(tgt_dir / Path(wav_path).name), wav, sample_rate=target_sr
            )

        logger.info(f"Resampling {split_dir} to {tgt_dir}:")
        Parallel(n_jobs=num_workers)(
            delayed(resample)(path) for path in tqdm(wav_paths)
        )
