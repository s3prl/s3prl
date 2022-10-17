import logging
from pathlib import Path

import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus import LibriSpeech
from s3prl.dataio.corpus import IEMOCAP

logger = logging.getLogger(__name__)

try:
    import torchaudio.datasets as datasets
except ModuleNotFoundError:
    test_torchaudio_dataset = False
else:
    test_torchaudio_dataset = True


@pytest.mark.corpus
def test_librispeech_with_torchaudio():
    if not test_torchaudio_dataset:
        return

    config = dotenv_values()
    dataset_root = Path(config["LibriSpeech"])
    corpus = LibriSpeech(dataset_root)
    train_data, valid_data, test_data = corpus.data_split

    def get_s3prl_paths_and_labels(data: dict):
        for key, value in data.items():
            yield (
                Path(value["wav_path"]).relative_to(dataset_root).as_posix(),
                value["transcription"],
            )

    train_set = datasets.LIBRISPEECH(dataset_root.parent, "train-clean-100")
    valid_set = datasets.LIBRISPEECH(dataset_root.parent, "dev-clean")
    test_set = datasets.LIBRISPEECH(dataset_root.parent, "test-clean")

    def get_torchaudio_paths_and_labels(dataset):
        for i in range(len(dataset)):
            path, sr, trans, spk, chapter, utt = dataset.get_metadata(i)
            yield (path, trans)

    def assert_corpus_parsing(data_s3prl: dict, data_torchaudio):
        data_s3prl = sorted(get_s3prl_paths_and_labels(data_s3prl))
        data_torchaudio = sorted(get_torchaudio_paths_and_labels(data_torchaudio))
        assert data_s3prl == data_torchaudio

    assert_corpus_parsing(train_data, train_set)
    assert_corpus_parsing(valid_data, valid_set)
    assert_corpus_parsing(test_data, test_set)


@pytest.mark.corpus
@pytest.mark.parametrize("session_id", [1, 2, 3, 4, 5])
def test_iemocap_with_torchaudio(session_id: int):
    config = dotenv_values()
    dataset_root = Path(config["IEMOCAP"])

    corpus = IEMOCAP(dataset_root)
    data_s3prl = corpus.get_whole_session(session_id)

    def get_s3prl_metadata(data: dict):
        for k, v in data.items():
            emotion = v["emotion"]
            if emotion in ["neu", "hap", "sad", "exc", "ang"]:
                if emotion == "exc":
                    emotion = "hap"
                yield (k, emotion)

    dataset = datasets.IEMOCAP(
        dataset_root.parent,
        sessions=[session_id],
    )

    def get_torchaudio_metadata(dataset):
        for i in range(len(dataset)):
            info = dataset.get_metadata(i)
            yield (info[2], info[3])

    info_s3prl = sorted(get_s3prl_metadata(data_s3prl))
    info_torchaudio = sorted(get_torchaudio_metadata(dataset))

    assert info_s3prl == info_torchaudio
