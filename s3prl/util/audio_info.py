import json
from pathlib import Path
from typing import List

import torchaudio
from joblib import Parallel, delayed
from tqdm import tqdm

_default_cache_dir = Path.home() / ".cache" / "s3prl" / "audio_info"


__all__ = [
    "get_cache_dir",
    "set_cache_dir",
    "get_audio_info",
]


def get_cache_dir():
    _default_cache_dir.mkdir(exist_ok=True, parents=True)
    return _default_cache_dir


def set_cache_dir(cache_dir: str):
    global _default_cache_dir
    _default_cache_dir = Path(cache_dir)


def get_audio_info(
    audio_paths: List[str],
    audio_ids: List[str],
    cache_dir: str = None,
    num_workers: int = 6,
) -> List[dict]:
    """
    Use :code:`torchaudio.info` to retrieve the metadata from audio paths.
    The retrieved metadata is cached in :code:`cache_dir`
    """

    cache_dir = cache_dir or get_cache_dir()
    cache_dir: Path = Path(cache_dir)

    def _get_info(audio_path: str, audio_id: str):
        cache_file = cache_dir / f"{audio_id}.json"
        if cache_file.is_file():
            with cache_file.open() as f:
                info = json.load(f)
                return info

        torchaudio.set_audio_backend("sox_io")
        torchaudio_info = torchaudio.info(audio_path)
        info = {
            "sample_rate": torchaudio_info.sample_rate,
            "num_frames": torchaudio_info.num_frames,
            "num_channels": torchaudio_info.num_channels,
            "bits_per_sample": torchaudio_info.bits_per_sample,
            "encoding": torchaudio_info.encoding,
        }
        return info

    infos = Parallel(n_jobs=num_workers)(
        delayed(_get_info)(path, idx)
        for path, idx in tqdm(zip(audio_paths, audio_ids), desc="Get audio metadata")
    )
    return infos
