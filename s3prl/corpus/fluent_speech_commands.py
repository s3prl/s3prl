from pathlib import Path

import pandas as pd

from s3prl import Container
from s3prl.util import registry

from .base import Corpus


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

        data_points = Container()
        data_points.add(self.train)
        data_points.add(self.valid)
        data_points.add(self.test)
        data_points = {key: self._parse_data(data) for key, data in data_points.items()}
        self._all_data = data_points

    @staticmethod
    def _get_unique_name(data_point):
        return Path(data_point["path"]).stem

    def _parse_data(self, data):
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
        import os
        import requests
        import tarfile
        
        assert os.path.exists(tgt_dir), "Target directory does not exist"

        def unzip_targz_then_delete(filepath: str):
            with tarfile.open(os.path.abspath(filepath)) as tar:
                tar.extractall(path=os.path.abspath(tgt_dir))
            os.remove(os.path.abspath(filepath))

        def download_from_url(url: str):
            filename = url.split("/")[-1].replace(" ", "_")
            filepath = os.path.join(tgt_dir, filename)

            r = requests.get(url, stream=True)
            if r.ok:
                logging.info(f"Saving {filename} to", os.path.abspath(filepath))
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024*10):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                logging.info(f"{filename} successfully downloaded")
                unzip_targz_then_delete(filepath)
            else:
                logging.info(f"Download failed: status code {r.status_code}\n{r.text}")

        if not (os.path.exists(os.path.join(os.path.abspath(tgt_dir), "fluent_speech_commands_dataset/wavs")) and 
                os.path.exists(os.path.join(os.path.abspath(tgt_dir), "fluent_speech_commands_dataset/data/speakers"))):
            download_from_url("http://140.112.21.28:9000/fluent.tar.gz")
        logging.info(f"Fluent speech commands dataset downloaded. Located at {os.path.abspath(tgt_dir)}/fluent_speech_commands_dataset/") 


@registry.put()
def fsc_for_multiple_classfication(dataset_root: str, n_jobs: int = 4):
    """
    Args:
        dataset_root: (str) The dataset root of fluent speech command

    Return:
        A :obj:`s3prl.base.container.Container` in

        .. code-block:: yaml

            train_data:
                data_id1:
                    wav_path: (str) waveform path
                    labels: (List[str]) The labels for action, object and location
                data_id2:

            valid_data:
                The same format as train_data

            test_data:
                The same format as valid_data
    """

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
