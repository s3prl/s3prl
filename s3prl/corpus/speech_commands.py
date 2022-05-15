import hashlib
import re
from pathlib import Path
from typing import List, Tuple, Union

from s3prl import Container, cache
from s3prl.base.output import Output

from .base import Corpus

CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "_unknown_",
    "_silence_",
]


class SpeechCommandsV1(Corpus):
    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        dataset_root = Path(dataset_root)
        train_dataset_root = dataset_root / "train"
        test_dataset_root = dataset_root / "test"

        train_list, valid_list = self.split_dataset(train_dataset_root)
        train_list = self.parse_train_valid_data_list(train_list, train_dataset_root)
        valid_list = self.parse_train_valid_data_list(valid_list, train_dataset_root)
        test_list = self.parse_test_data_list(test_dataset_root)

        self.train = self.list_to_dict(train_list)
        self.valid = self.list_to_dict(valid_list)
        self.test = self.list_to_dict(test_list)

        self._data = Container()
        self._data.add(self.train)
        self._data.update(self.valid, override=True)  # background noises are duplicated
        self._data.add(self.test)

    @staticmethod
    @cache()
    def split_dataset(
        root_dir: Union[str, Path], max_uttr_per_class=2**27 - 1
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Split Speech Commands into 3 set.

        Args:
            root_dir: speech commands dataset root dir
            max_uttr_per_class: predefined value in the original paper

        Return:
            train_list: [(class_name, audio_path), ...]
            valid_list: as above
        """
        train_list, valid_list = [], []

        for entry in Path(root_dir).iterdir():
            if not entry.is_dir() or entry.name == "_background_noise_":
                continue

            for audio_path in entry.glob("*.wav"):
                speaker_hashed = re.sub(r"_nohash_.*$", "", audio_path.name)
                hashed_again = hashlib.sha1(speaker_hashed.encode("utf-8")).hexdigest()
                percentage_hash = (int(hashed_again, 16) % (max_uttr_per_class + 1)) * (
                    100.0 / max_uttr_per_class
                )

                if percentage_hash < 10:
                    valid_list.append((entry.name, audio_path))
                elif percentage_hash < 20:
                    pass  # testing set is discarded
                else:
                    train_list.append((entry.name, audio_path))

        return train_list, valid_list

    @staticmethod
    @cache()
    def parse_train_valid_data_list(data_list, train_dataset_root: Path):
        data = [
            (class_name, audio_path)
            if class_name in CLASSES
            else ("_unknown_", audio_path)
            for class_name, audio_path in data_list
        ]
        data += [
            ("_silence_", audio_path)
            for audio_path in Path(train_dataset_root, "_background_noise_").glob(
                "*.wav"
            )
        ]
        return data

    @staticmethod
    @cache()
    def parse_test_data_list(test_dataset_root: Path):
        data = [
            (class_dir.name, audio_path)
            for class_dir in Path(test_dataset_root).iterdir()
            if class_dir.is_dir()
            for audio_path in class_dir.glob("*.wav")
        ]
        return data

    @staticmethod
    def path_to_unique_name(path: str):
        return "/".join(Path(path).parts[-2:])

    @classmethod
    def list_to_dict(cls, data_list):
        data = Container(
            {
                cls.path_to_unique_name(audio_path): {
                    "wav_path": audio_path,
                    "class_name": class_name,
                }
                for class_name, audio_path in data_list
            }
        )
        return data

    @property
    def all_data(self):
        """
        Return:
            Container: id (str)
                wav_path (str)
                class_name (str)
        """
        return self._data

    @property
    def data_split_ids(self):
        return list(self.train.keys()), list(self.valid.keys()), list(self.test.keys())


class SpeechCommandsV1ForSUPERB(SpeechCommandsV1):
    """
    TODO: Weighted Sampling!
    """

    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        super().__init__(dataset_root, n_jobs)

    @staticmethod
    def format_fields(data: dict):
        formated_data = Container(
            {
                key: {
                    "wav_path": value.wav_path,
                    "label": value.class_name,
                }
                for key, value in data.items()
            }
        )
        return formated_data

    def __call__(self):
        train_data, valid_data, test_data = self.data_split
        return Output(
            train_data=self.format_fields(train_data),
            valid_data=self.format_fields(valid_data),
            test_data=self.format_fields(test_data),
        )
