"""
Parse the IEMOCAP corpus

Authors:
  * Leo 2022
  * Cheng Liang 2022
"""

import logging
import re
from copy import deepcopy
from pathlib import Path

from librosa.util import find_files

from .base import Corpus

IEMOCAP_SESSION_NUM = 5
LABEL_DIR_PATH = "dialog/EmoEvaluation"
WAV_DIR_PATH = "sentences/wav"

logger = logging.getLogger(__name__)

__all__ = [
    "IEMOCAP",
]


class IEMOCAP(Corpus):
    """
    Parse the IEMOCAP dataset

    Args:
        dataset_root: (str) The dataset root of IEMOCAP
    """

    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        self.dataset_root = Path(dataset_root)
        self.sessions = [
            (self._preprocess_single_session(self.dataset_root, session_id))
            for session_id in range(1, IEMOCAP_SESSION_NUM + 1)
        ]

        self._all_data = dict()
        for session in self.sessions:
            self._all_data.update(session["improvised"])
            self._all_data.update(session["scripted"])

    @staticmethod
    def _preprocess_single_session(dataset_root: Path, session_id: int):
        data = dict(
            improvised={},
            scripted={},
        )

        session_dir = dataset_root / f"Session{session_id}"
        label_dir = session_dir / LABEL_DIR_PATH
        wav_root_dir = session_dir / WAV_DIR_PATH
        wav_paths = find_files(wav_root_dir)
        for wav_path in wav_paths:
            wav_path = Path(wav_path)
            spk_and_act_and_scene = wav_path.parts[-2]
            label_file = label_dir / f"{spk_and_act_and_scene}.txt"
            with label_file.open() as file:
                content = file.read()
            result = re.search(rf"{str(wav_path.stem)}\t(.+)\t", content)
            speaker = spk_and_act_and_scene.split("_")[0]
            act = "improvised" if "impro" in spk_and_act_and_scene else "scripted"
            emotion = result.groups()[0]
            unique_id = wav_path.stem

            data[act][unique_id] = dict(
                wav_path=str(wav_path),
                speaker=speaker,
                act=act,
                emotion=emotion,
                session_id=session_id,
            )

        return data

    @property
    def all_data(self):
        """
        Return:
            dict

            all the data points of IEMOCAP in the format of

            .. code-block:: yaml

                data_id1:
                    wav_path (str): The waveform path
                    speaker (str): The speaker name
                    act (str): improvised / scripted
                    emotion (str): The emotion label
                    session_id (int): The session

                data_id2:
                    ...
        """
        return deepcopy(self._all_data.copy())

    def get_whole_session(self, session_id: int):
        """
        Args:
            session_id (int): The session index selected from 1, 2, 3, 4, 5

        Return:
            dict

            data points in a single session (containing improvised and scripted recordings) in the
            same format as :obj:`all_data`
        """
        output = dict()
        output.update(self.get_session_with_act(session_id, "improvised"))
        output.update(self.get_session_with_act(session_id, "scripted"))
        return deepcopy(output)

    def get_session_with_act(self, session_id: int, act: str):
        """
        Args:
            session_id (int): The session index selected from 1, 2, 3, 4, 5
            act (str): 'improvised' or 'scripted'

        Return:
            :obj:`s3prl.base.container.Container`

            data points in a single session with a specific act (either improvised or scripted) in the
            same format as :obj:`all_data`
        """
        assert act in ["improvised", "scripted"]
        return deepcopy(self.sessions[session_id - 1][act])

    @classmethod
    def download_dataset(cls, tgt_dir: str) -> None:
        import os
        import tarfile

        import requests

        assert os.path.exists(
            os.path.abspath(tgt_dir)
        ), "Target directory does not exist"

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

        if not os.path.exists(os.path.join(os.path.abspath(tgt_dir), "IEMOCAP/")):
            download_from_url("http://140.112.21.28:9000/IEMOCAP.tar.gz")
        logger.info(
            f"IEMOCAP dataset downloaded. Located at {os.path.abspath(tgt_dir)}/IEMOCAP/"
        )
