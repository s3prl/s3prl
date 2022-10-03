"""
Parse VoxCeleb1 corpus for classification

Authors:
  * Leo 2022
  * Cheng Liang 2022
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List

from filelock import FileLock
from joblib import Parallel, delayed
from tqdm import tqdm

from .base import Corpus

logger = logging.getLogger(__name__)

SPLIT_FILE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
CACHE_ROOT = Path.home() / ".cache" / "s3prl"

__all__ = [
    "VoxCeleb1SID",
]


class VoxCeleb1SID(Corpus):
    def __init__(
        self, dataset_root: str, n_jobs: int = 4, cache_root: str = CACHE_ROOT
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()

        uid2split = self._get_standard_usage(self.dataset_root, cache_root)
        self._split2uids = defaultdict(list)
        for uid, split in uid2split.items():
            self._split2uids[split].append(Path(uid.replace("/", "-")).stem)

        uid2wavpath = self._find_wavs_with_uids(
            self.dataset_root, sorted(uid2split.keys()), n_jobs=n_jobs
        )
        self._data = {
            Path(uid.replace("/", "-")).stem: {
                "wav_path": uid2wavpath[uid],
                "label": self._build_label(uid),
            }
            for uid in uid2split.keys()
        }

    @property
    def all_data(self):
        return self._data

    @property
    def data_split_ids(self):
        return (
            self._split2uids["train"],
            self._split2uids["valid"],
            self._split2uids["test"],
        )

    @staticmethod
    def _get_standard_usage(dataset_root: Path, cache_root: Path):
        split_filename = SPLIT_FILE_URL.split("/")[-1]
        split_filepath = Path(cache_root) / split_filename
        if not split_filepath.is_file():
            with FileLock(str(split_filepath) + ".lock"):
                os.system(f"wget {SPLIT_FILE_URL} -O {str(split_filepath)}")
        standard_usage = [
            line.strip().split(" ") for line in open(split_filepath, "r").readlines()
        ]

        def code2split(code: int):
            splits = ["train", "valid", "test"]
            return splits[code - 1]

        standard_usage = {uid: code2split(int(split)) for split, uid in standard_usage}
        return standard_usage

    @staticmethod
    def _find_wavs_with_uids(dataset_root, uids, n_jobs=4):
        def find_wav_with_uid(uid):
            found_wavs = list(dataset_root.glob(f"*/wav/{uid}"))
            assert len(found_wavs) == 1
            return uid, found_wavs[0]

        uids_with_wavs = Parallel(n_jobs=n_jobs)(
            delayed(find_wav_with_uid)(uid) for uid in tqdm(uids, desc="Search wavs")
        )
        uids2wav = {uid: wav for uid, wav in uids_with_wavs}
        return uids2wav

    @staticmethod
    def _build_label(uid):
        id_string = uid.split("/")[0]
        label = f"speaker_{int(id_string[2:]) - 10001}"
        return label

    @classmethod
    def download_dataset(
        cls, target_dir: str, splits: List[str] = ["dev", "test"]
    ) -> None:
        tgt_dir = os.path.abspath(target_dir)
        assert os.path.exists(tgt_dir), "Target directory does not exist"

        from zipfile import ZipFile

        import requests

        def unzip_then_delete(filepath: str, split: str):
            assert os.path.exists(filepath), "File not found!"

            with ZipFile(filepath) as zipf:
                zipf.extractall(path=os.path.join(tgt_dir, "Voxceleb1", split))
            os.remove(os.path.abspath(filepath))

        def download_from_url(url: str, split: str):
            filename = url.split("/")[-1].replace(" ", "_")
            filepath = os.path.join(tgt_dir, filename)

            r = requests.get(url, stream=True)
            if r.ok:
                logger.info(f"Saving {filename} to", filepath)
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024 * 10):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                logger.info(f"{filename} successfully downloaded")
            else:
                logger.info(f"Download failed: status code {r.status_code}\n{r.text}")

            return filepath

        def download_dev():
            partpaths = []

            for part in ["a", "b", "c", "d"]:
                if os.path.exists(os.path.join(tgt_dir, f"vox1_dev_wav_parta{part}")):
                    logger.info(f"vox1_dev_wav_parta{part} exists, skip donwload")
                    partpaths.append(os.path.join(tgt_dir, f"vox1_dev_wav_parta{part}"))
                    continue
                fp = download_from_url(
                    f"https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_parta{part}",
                    "dev",
                )
                partpaths.append(fp)

            zippath = os.path.join(tgt_dir, "vox1_dev_wav.zip")
            with open(zippath, "wb") as outfile:
                for f in partpaths:
                    with open(f, "rb") as infile:
                        for line in infile:
                            outfile.write(line)

            for f in partpaths:
                os.remove(f)
            unzip_then_delete(zippath, "dev")

        for split in splits:
            if not os.path.exists(os.path.join(tgt_dir, "Voxceleb1/" + split + "/wav")):
                if split == "dev":
                    download_dev()
                else:
                    filepath = download_from_url(
                        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip",
                        "test",
                    )
                    unzip_then_delete(filepath, "test")

        logger.info(f"Voxceleb1 dataset downloaded. Located at {tgt_dir}/Voxceleb1/")
