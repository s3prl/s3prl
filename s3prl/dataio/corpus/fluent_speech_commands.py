"""
Parse the Fluent Speech Command corpus

Authors:
  * Leo 2022
  * Cheng Liang 2022
"""

import logging
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from .base import Corpus

logger = logging.getLogger(__name__)

__all__ = [
    "FluentSpeechCommands",
]


class FluentSpeechCommands(Corpus):
    """
    Parse the Fluent Speech Command dataset

    Args:
        dataset_root: (str) The dataset root of Fluent Speech Command
    """

    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        self.dataset_root = Path(dataset_root)
        self.train = self.dataframe_to_datapoints(
            pd.read_csv(self.dataset_root / "data" / "train_data.csv"),
            self._get_unique_name,
        )
        self.valid = self.dataframe_to_datapoints(
            pd.read_csv(self.dataset_root / "data" / "valid_data.csv"),
            self._get_unique_name,
        )
        self.test = self.dataframe_to_datapoints(
            pd.read_csv(self.dataset_root / "data" / "test_data.csv"),
            self._get_unique_name,
        )

        data_points = OrderedDict()
        data_points.update(self.train)
        data_points.update(self.valid)
        data_points.update(self.test)
        data_points = {key: self._parse_data(data) for key, data in data_points.items()}
        self._all_data = data_points

    @staticmethod
    def _get_unique_name(data_point):
        return Path(data_point["path"]).stem

    def _parse_data(self, data):
        return dict(
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
        Return all the data points in a dict of the format

        .. code-block:: yaml

            data_id1:
                path: (str) The waveform path
                speakerId: (str) The speaker name
                transcription: (str) The transcription
                action: (str) The action
                object: (str) The action's targeting object
                location: (str) The location where the action happens

            data_id2:
                ...
        """
        return self._all_data

    @property
    def data_split(self):
        """
        Return a list:

        :code:`train_data`, :code:`valid_data`, :code:`test_data`

        each is a dict following the format specified in :obj:`all_data`
        """
        return super().data_split

    @property
    def data_split_ids(self):
        """
        Return a list:

        :code:`train_ids`, :code:`valid_ids`, :code:`test_ids`

        Each is a list containing data_ids. data_ids can be used as the key to access the :obj:`all_data`
        """
        return list(self.train.keys()), list(self.valid.keys()), list(self.test.keys())

    @classmethod
    def download_dataset(cls, tgt_dir: str) -> None:
        """
        Download and unzip the dataset to :code:`tgt_dir`/fluent_speech_commands_dataset

        Args:
            tgt_dir (str): The root directory containing many different datasets
        """
        import os
        import tarfile

        import requests

        tgt_dir = Path(tgt_dir)
        tgt_dir.mkdir(exists_ok=True, parents=True)

        def unzip_targz_then_delete(filepath: str):
            with tarfile.open(os.path.abspath(filepath)) as tar:
                tar.extractall(path=os.path.abspath(tgt_dir))
            os.remove(os.path.abspath(filepath))

        def download_from_url(url: str):
            filename = url.split("/")[-1].replace(" ", "_")
            filepath = os.path.join(tgt_dir, filename)

            r = requests.get(url, stream=True)
            if r.ok:
                logger.info(f"Saving {filename} to", os.path.abspath(filepath))
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

        if not (
            os.path.exists(
                os.path.join(
                    os.path.abspath(tgt_dir), "fluent_speech_commands_dataset/wavs"
                )
            )
            and os.path.exists(
                os.path.join(
                    os.path.abspath(tgt_dir),
                    "fluent_speech_commands_dataset/data/speakers",
                )
            )
        ):
            download_from_url("http://140.112.21.28:9000/fluent.tar.gz")
        logger.info(
            f"Fluent speech commands dataset downloaded. Located at {os.path.abspath(tgt_dir)}/fluent_speech_commands_dataset/"
        )
