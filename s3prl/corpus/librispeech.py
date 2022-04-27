import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from joblib import Parallel, delayed

from s3prl import Container, Output, cache

from .base import Corpus

LIBRI_SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
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


class LibriSpeech(Corpus):
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

        self._data = Container()
        self._data.add(self.train)
        self._data.add(self.valid)
        self._data.add(self.test)

    def get_corpus_splits(self, splits: List[str]):
        return self._data_to_dict(self.data_dict, splits)

    @property
    def all_data(self):
        return self._data

    @property
    def data_split_ids(self):
        return (
            list(self.train.keys()),
            list(self.valid.keys()),
            list(self.test.keys()),
        )

    @staticmethod
    @cache()
    def _collect_data(
        dataset_root: str, splits: List[str], n_jobs: int = 4
    ) -> Dict[str, Dict[str, List[Any]]]:

        data_dict = {}
        for split in splits:
            split_dir = os.path.join(dataset_root, split)
            if not os.path.exists(split_dir):
                logging.info(f"Split {split} is not downloaded. Skip data collection.")
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
            }

        return data_dict

    @staticmethod
    def _data_to_dict(
        data_dict: Dict[str, Dict[str, List[Any]]], splits: List[str]
    ) -> Container:
        data = Container(
            {
                name: {
                    "wav_path": data_dict[split]["wav_list"][i],
                    "transcription": data_dict[split]["text_list"][i],
                    "speaker": data_dict[split]["spkr_list"][i],
                    "corpus_split": split,
                }
                for split in splits
                for i, name in enumerate(data_dict[split]["name_list"])
            }
        )
        return data


class LibriSpeechForSpeech2Text(LibriSpeech):
    def __init__(
        self,
        dataset_root: str,
        n_jobs: int = 4,
        train_split: List[str] = ["train-clean-100"],
        valid_split: List[str] = ["dev-clean"],
        test_split: List[str] = ["test-clean"],
    ) -> None:
        super().__init__(dataset_root, n_jobs, train_split, valid_split, test_split)

    def __call__(self):
        train_data, valid_data, test_data = self.data_split
        return Output(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
        )


class LibriSpeechForSUPERB(LibriSpeechForSpeech2Text):
    def __init__(
        self,
        dataset_root: str,
        n_jobs: int = 4,
    ) -> None:
        super().__init__(
            dataset_root, n_jobs, ["train-clean-100"], ["dev-clean"], ["test-clean"]
        )
