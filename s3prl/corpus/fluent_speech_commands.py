from pathlib import Path

import pandas as pd

from s3prl import Container
from s3prl.util import registry

from .base import Corpus


class FluentSpeechCommands(Corpus):
    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        self.dataset_root = Path(dataset_root)
        self.train = self.dataframe_to_datapoints(
            pd.read_csv(self.dataset_root / "data" / "train_data.csv"),
            self.get_unique_name,
        )
        self.valid = self.dataframe_to_datapoints(
            pd.read_csv(self.dataset_root / "data" / "valid_data.csv"),
            self.get_unique_name,
        )
        self.test = self.dataframe_to_datapoints(
            pd.read_csv(self.dataset_root / "data" / "test_data.csv"),
            self.get_unique_name,
        )

        data_points = Container()
        data_points.add(self.train)
        data_points.add(self.valid)
        data_points.add(self.test)
        data_points = {key: self.parse_data(data) for key, data in data_points.items()}
        self._all_data = data_points

    @staticmethod
    def get_unique_name(data_point):
        return Path(data_point["path"]).stem

    def parse_data(self, data):
        return Container(
            path=self.dataset_root / data["path"],
            speakerId=data["speakerId"],
            transcription=data["transcription"],
            action=data["action"],
            object=data["object"],
            location=data["location"],
        )

    @property
    def all_data(self):
        """
        Return:
            Container: id (str)
                path (str)
                speakerId (str)
                transcription (str)
                action (str)
                object (str)
                location (str)
        """
        return self._all_data

    @property
    def data_split_ids(self):
        return list(self.train.keys()), list(self.valid.keys()), list(self.test.keys())


@registry.put()
def fsc_for_multiple_classfication(dataset_root: str, n_jobs: int = 4):
    def format_fields(data_points):
        return {
            key: dict(
                wav_path=value.path,
                labels=[value.action, value.object, value.location],
            )
            for key, value in data_points.items()
        }

    corpus = FluentSpeechCommands(dataset_root, n_jobs)
    train_data, valid_data, test_data = corpus.data_split
    return Container(
        train_data=format_fields(train_data),
        valid_data=format_fields(valid_data),
        test_data=format_fields(test_data),
    )
