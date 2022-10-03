"""
Parse the LibriLight corpus

Authors:
  * Heng-Jui Chang 2022
"""

import logging
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from joblib import Parallel, delayed

from s3prl.util.download import _urls_to_filepaths

from .base import Corpus

LIBRILIGHT_SPLITS = [
    "10h",
    "1h",
    "10m-fold0",
    "10m-fold1",
    "10m-fold2",
    "10m-fold3",
    "10m-fold4",
    "10m-fold5",
]
LIBRISPEECH_SPKR_INFO = (
    "https://huggingface.co/datasets/s3prl/librispeech_metadata/raw/main/SPEAKERS.TXT"
)


__all__ = [
    "LibriLight",
]


def read_text(file: Path) -> str:
    src_file = "-".join(str(file).split("-")[:-1]) + ".trans.txt"
    idx = file.stem.replace(".flac", "")

    with open(src_file, "r") as fp:
        for line in fp:
            if idx == line.split(" ")[0]:
                return line[:-1].split(" ", 1)[1]

    logging.warning(f"Transcription of {file} not found!")


def check_no_repeat(splits: List[str]) -> bool:
    count = defaultdict(int)
    for split in splits:
        count[split] += 1

    repeated = ""
    for key, val in count.items():
        if val > 1:
            repeated += f" {key} ({val} times)"

    if len(repeated) != 0:
        logging.warning(
            f"Found repeated splits in corpus: {repeated}, which might cause unexpected behaviors."
        )
        return False

    return True


def _parse_spk_to_gender(speaker_file: Path) -> dict:
    speaker_file = Path(speaker_file)
    with speaker_file.open() as file:
        lines = [line.strip() for line in file.readlines()]
    for line_id in range(len(lines)):
        line = lines[line_id]
        if "SEX" in line and "SUBSET" in line and "MINUTES" in line and "NAME" in line:
            break

    line_id += 1  # first line with speaker info
    spk2gender = {}
    for line_id in range(line_id, len(lines)):
        line = lines[line_id]
        line = re.sub("\t+", " ", line)
        line = re.sub(" +", " ", line)
        parts = line.split("|", maxsplit=4)
        ID, SEX, SUBSET, MINUTES, NAME = parts
        spk2gender[int(ID)] = SEX.strip()
    return spk2gender


class LibriLight(Corpus):
    def __init__(
        self,
        dataset_root: str,
        n_jobs: int = 4,
        train_split: str = "10m-fold0",
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.train_split = train_split

        if train_split == "10h":
            roots = [self.dataset_root / "1h", self.dataset_root / "9h"]
        elif train_split == "1h":
            roots = [self.dataset_root / "1h"]
        elif train_split.startswith("10m"):
            fold_id = int(train_split.split("-")[-1].split("fold")[-1])
            roots = [self.dataset_root / "1h" / str(fold_id)]
        else:
            raise ValueError(f"Unsupported split: {train_split}")

        self._data = self._collect_data(roots, n_jobs)

    @classmethod
    def download_dataset(cls, dataset_root: str):
        Path(dataset_root).mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                "wget",
                "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz",
                "-O",
                str(Path(dataset_root) / "librispeech_finetuning.tgz"),
            ]
        )
        subprocess.check_call(
            ["tar", "zxvf", "librispeech_finetuning.tgz", "-C", str(Path(dataset_root))]
        )

    @property
    def all_data(self):
        return self._data

    @staticmethod
    def _collect_data(
        roots: List[Path], n_jobs: int = 4
    ) -> Dict[str, Dict[str, List[Any]]]:
        spkr_file = _urls_to_filepaths(LIBRISPEECH_SPKR_INFO)
        spkr2gender = _parse_spk_to_gender(Path(spkr_file).resolve())
        data_dict = {}
        for split_dir in roots:
            if not os.path.exists(split_dir):
                logging.info(
                    f"Split {split_dir} is not downloaded. Skip data collection."
                )
                continue

            wav_list = list(Path(split_dir).rglob("*.flac"))
            name_list = [file.stem.replace(".flac", "") for file in wav_list]
            text_list = Parallel(n_jobs=n_jobs)(
                delayed(read_text)(file) for file in wav_list
            )
            spkr_list = [int(name.split("-")[0]) for name in name_list]

            for wav_id in range(len(wav_list)):
                wav = Path(wav_list[wav_id])
                data_dict[wav.stem] = {
                    "wav_path": str(wav.resolve()),
                    "transcription": text_list[wav_id],
                    "speaker": spkr_list[wav_id],
                    "gender": spkr2gender[spkr_list[wav_id]],
                }
        return data_dict
