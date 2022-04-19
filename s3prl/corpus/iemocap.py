import re
from pathlib import Path
from librosa.util import find_files

from s3prl import Container, cache
from .base import Corpus

IEMOCAP_SESSION_NUM = 5
LABEL_DIR_PATH = "dialog/EmoEvaluation"
WAV_DIR_PATH = "sentences/wav"


class IEMOCAP(Corpus):
    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        self.dataset_root = Path(dataset_root)
        self.sessions = [
            Container(self.preprocess_single_session(self.dataset_root, session_id))
            for session_id in range(1, IEMOCAP_SESSION_NUM + 1)
        ]

        self._all_data = Container()
        for session in self.sessions:
            self._all_data.add(session.improvised)
            self._all_data.add(session.scripted)

    @staticmethod
    @cache()
    def preprocess_single_session(dataset_root: Path, session_id: int):
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
            result = re.search(fr"{str(wav_path.stem)}\t(.+)\t", content)
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
            Container: id (str)
                wav_path (str)
                speaker (str)
                act (str), improvised / scripted
                emotion (str)
                session_id (int)
        """
        return Container(self._all_data)

    def get_whole_session(self, session_id: int):
        output = Container()
        output.add(self.get_session_with_act(session_id, "improvised"))
        output.add(self.get_session_with_act(session_id, "scripted"))
        return Container(output)

    def get_session_with_act(self, session_id: int, act: str):
        assert act in ["improvised", "scripted"]
        return Container(self.sessions[session_id - 1][act])


class IEMOCAPForSUPERB(IEMOCAP):
    def __init__(
        self, dataset_root: str, test_session: int = 1, n_jobs: int = 4
    ) -> None:
        super().__init__(dataset_root, n_jobs)
        self.test_session = test_session

    @staticmethod
    def format_fields(data_points):
        return {
            key: dict(
                wav_path=value.wav_path,
                label=value.emotion,
            )
            for key, value in data_points.items()
        }

    @staticmethod
    def filter_data(data: Container):
        for key in list(data.keys()):
            data_point = data[key]
            if data_point.emotion not in ["neu", "hap", "ang", "sad", "exc"]:
                del data[key]
            if data_point.emotion == "exc":
                data_point.emotion = "hap"
        return data

    def __call__(self):
        valid_session = (self.test_session + 1) % IEMOCAP_SESSION_NUM
        train_sessions = [
            s + 1
            for s in list(range(IEMOCAP_SESSION_NUM))
            if s + 1 not in [valid_session, self.test_session]
        ]
        train_data = Container()
        for session_id in train_sessions:
            train_data.add(self.get_whole_session(session_id))
        valid_data = self.get_whole_session(valid_session)
        test_data = self.get_whole_session(self.test_session)

        return Container(
            train_data=self.format_fields(self.filter_data(train_data)),
            valid_data=self.format_fields(self.filter_data(valid_data)),
            test_data=self.format_fields(self.filter_data(test_data)),
        )
