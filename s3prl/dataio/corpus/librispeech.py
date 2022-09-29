"""
Parse the LibriSpeech corpus

Authors:
  * Heng-Jui Chang 2022
"""

import logging
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from joblib import Parallel, delayed

from .base import Corpus

logger = logging.getLogger(__name__)

LIBRI_SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

__all__ = [
    "LibriSpeech",
]


def read_text(file: Path) -> str:
    src_file = "-".join(str(file).split("-")[:-1]) + ".trans.txt"
    idx = file.stem.replace(".flac", "")

    with open(src_file, "r") as fp:
        for line in fp:
            if idx == line.split(" ")[0]:
                return line[:-1].split(" ", 1)[1]

    logger.warning(f"Transcription of {file} not found!")


def check_no_repeat(splits: List[str]) -> bool:
    count = defaultdict(int)
    for split in splits:
        count[split] += 1

    repeated = ""
    for key, val in count.items():
        if val > 1:
            repeated += f" {key} ({val} times)"

    if len(repeated) != 0:
        logger.warning(
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


class LibriSpeech(Corpus):
    """LibriSpeech Corpus
    Link: https://www.openslr.org/12

    Args:
        dataset_root (str): Path to LibriSpeech corpus directory.
        n_jobs (int, optional): Number of jobs. Defaults to 4.
        train_split (List[str], optional): Training splits. Defaults to ["train-clean-100"].
        valid_split (List[str], optional): Validation splits. Defaults to ["dev-clean"].
        test_split (List[str], optional): Testing splits. Defaults to ["test-clean"].
    """

    def __init__(
        self,
        dataset_root: str,
        n_jobs: int = 4,
        train_split: List[str] = ["train-clean-100"],
        valid_split: List[str] = ["dev-clean"],
        test_split: List[str] = ["test-clean"],
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        self.all_splits = train_split + valid_split + test_split
        assert check_no_repeat(self.all_splits)

        self.data_dict = self._collect_data(dataset_root, self.all_splits, n_jobs)
        self.train = self._data_to_dict(self.data_dict, train_split)
        self.valid = self._data_to_dict(self.data_dict, valid_split)
        self.test = self._data_to_dict(self.data_dict, test_split)

        self._data = OrderedDict()
        self._data.update(self.train)
        self._data.update(self.valid)
        self._data.update(self.test)

    def get_corpus_splits(self, splits: List[str]):
        return self._data_to_dict(self.data_dict, splits)

    @property
    def all_data(self):
        """
        Return all the data points in a dict of the format

        .. code-block:: yaml

            data_id1:
                wav_path: (str) The waveform path
                transcription: (str) The transcription
                speaker: (str) The speaker name
                gender: (str) The speaker's gender
                corpus_split: (str) The split of corpus this sample belongs to

            data_id2:
                ...
        """
        return self._data

    @property
    def data_split_ids(self):
        return (
            list(self.train.keys()),
            list(self.valid.keys()),
            list(self.test.keys()),
        )

    @staticmethod
    def _collect_data(
        dataset_root: str, splits: List[str], n_jobs: int = 4
    ) -> Dict[str, Dict[str, List[Any]]]:

        spkr2gender = _parse_spk_to_gender(Path(dataset_root) / "SPEAKERS.TXT")
        data_dict = {}
        for split in splits:
            split_dir = os.path.join(dataset_root, split)
            if not os.path.exists(split_dir):
                logger.info(f"Split {split} is not downloaded. Skip data collection.")
                continue

            wav_list = list(Path(split_dir).rglob("*.flac"))
            name_list = [file.stem.replace(".flac", "") for file in wav_list]
            text_list = Parallel(n_jobs=n_jobs)(
                delayed(read_text)(file) for file in wav_list
            )
            spkr_list = [int(name.split("-")[0]) for name in name_list]

            wav_list, name_list, text_list, spkr_list = zip(
                *[
                    (wav, name, text, spkr)
                    for (wav, name, text, spkr) in sorted(
                        zip(wav_list, name_list, text_list, spkr_list),
                        key=lambda x: x[1],
                    )
                ]
            )
            data_dict[split] = {
                "name_list": list(name_list),
                "wav_list": list(wav_list),
                "text_list": list(text_list),
                "spkr_list": list(spkr_list),
                "gender_list": [spkr2gender[spkr] for spkr in spkr_list],
            }

        return data_dict

    @staticmethod
    def _data_to_dict(
        data_dict: Dict[str, Dict[str, List[Any]]], splits: List[str]
    ) -> dict():
        data = dict(
            {
                name: {
                    "wav_path": data_dict[split]["wav_list"][i],
                    "transcription": data_dict[split]["text_list"][i],
                    "speaker": data_dict[split]["spkr_list"][i],
                    "gender": data_dict[split]["gender_list"][i],
                    "corpus_split": split,
                }
                for split in splits
                for i, name in enumerate(data_dict[split]["name_list"])
            }
        )
        return data

    @classmethod
    def download_dataset(
        cls,
        target_dir: str,
        splits: List[str] = ["train-clean-100", "dev-clean", "test-clean"],
    ) -> None:
        import os
        import tarfile

        import requests

        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)

        def unzip_targz_then_delete(filepath: str):
            with tarfile.open(os.path.abspath(filepath)) as tar:
                tar.extractall(path=os.path.abspath(target_dir))
            os.remove(os.path.abspath(filepath))

        def download_from_url(url: str):
            filename = url.split("/")[-1].replace(" ", "_")
            filepath = os.path.join(target_dir, filename)

            r = requests.get(url, stream=True)
            if r.ok:
                logger.info(f"Saving {filename} to {os.path.abspath(filepath)}")
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024 * 10):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                logger.info(f"{filename} successfully downloaded")
                unzip_targz_then_delete(filepath)
            else:
                logger.info(f"Download failed: status code {r.status_code}\n{r.text}")

        for split in splits:
            if not os.path.exists(
                os.path.join(os.path.abspath(target_dir), "Librispeech/" + split)
            ):
                download_from_url(
                    "https://www.openslr.org/resources/12/" + split + ".tar.gz"
                )
        logger.info(
            ", ".join(splits)
            + f"downloaded. Located at {os.path.abspath(target_dir)}/Librispeech/"
        )
