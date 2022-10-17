import logging
from pathlib import Path

import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus import (
    IEMOCAP,
    SNIPS,
    FluentSpeechCommands,
    LibriSpeech,
    Quesst14,
    VoxCeleb1SID,
    VoxCeleb1SV,
)
from s3prl.problem.asr.superb_sf import TEST_SPEAKERS, TRAIN_SPEAKERS, VALID_SPEAKERS

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


@pytest.mark.corpus
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_fluent_with_torchaudio(split: str):
    config = dotenv_values()
    dataset_root = Path(config["FluentSpeechCommands"])

    corpus = FluentSpeechCommands(dataset_root)
    train_data, valid_data, test_data = corpus.data_split
    if split == "train":
        s3prl_data = train_data
    elif split == "valid":
        s3prl_data = valid_data
    elif split == "test":
        s3prl_data = test_data

    def get_s3prl_metadata(data: dict):
        for k, v in data.items():
            yield (k, v["action"], v["object"], v["location"])

    dataset = datasets.FluentSpeechCommands(dataset_root.parent, split)

    def get_torchaudio_metadata(dataset):
        for i in range(len(dataset)):
            info = dataset.get_metadata(i)
            yield (info[2], info[5], info[6], info[7])

    info_s3prl = sorted(get_s3prl_metadata(s3prl_data))
    info_torchaudio = sorted(get_torchaudio_metadata(dataset))
    assert info_s3prl == info_torchaudio


@pytest.mark.corpus
@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_voxceleb1_with_torchaudio(split: str):
    config = dotenv_values()
    dataset_root = Path(config["VoxCeleb1"])

    corpus = VoxCeleb1SID(dataset_root)
    train_data, valid_data, test_data = corpus.data_split
    if split == "train":
        s3prl_data = train_data
    elif split == "dev":
        s3prl_data = valid_data
    elif split == "test":
        s3prl_data = test_data

    def get_s3prl_metadata(data: dict):
        for k, v in data.items():
            yield (k, v["label"])

    dataset = datasets.VoxCeleb1Identification(dataset_root, split)

    def get_torchaudio_metadata(dataset):
        for i in range(len(dataset)):
            info = dataset.get_metadata(i)
            yield (info[3], str(info[2]))

    info_s3prl = sorted(get_s3prl_metadata(s3prl_data))
    info_torchaudio = sorted(get_torchaudio_metadata(dataset))
    assert info_s3prl == info_torchaudio


@pytest.mark.corpus
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_snips_with_torchaudio(split: str):
    config = dotenv_values()
    dataset_root = Path(config["SNIPS"])

    corpus = SNIPS(dataset_root, TRAIN_SPEAKERS, VALID_SPEAKERS, TEST_SPEAKERS)
    train_data, valid_data, test_data = corpus.data_split

    split_data_map = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
    }

    def get_s3prl_metadata(data: dict):
        for k, v in data.items():
            yield (k, v["transcription"], v["iob"], v["intent"])

    split_spk_map = {
        "train": TRAIN_SPEAKERS,
        "valid": VALID_SPEAKERS,
        "test": TEST_SPEAKERS,
    }
    dataset = datasets.Snips(dataset_root.parent, split, split_spk_map[split], "wav")

    def get_torchaudio_metadata(dataset):
        for i in range(len(dataset)):
            info = dataset.get_metadata(i)
            yield (Path(info[0]).stem, info[2], info[3], info[4])

    info_s3prl = sorted(get_s3prl_metadata(split_data_map[split]))
    info_torchaudio = sorted(get_torchaudio_metadata(dataset))
    assert info_s3prl == info_torchaudio


@pytest.mark.corpus
@pytest.mark.parametrize("split", ["docs", "dev", "eval"])
def test_quesst14_with_torchaudio(split: str):
    config = dotenv_values()
    dataset_root = Path(config["Quesst14"])

    corpus = Quesst14(dataset_root)
    s3prl_data = {
        "docs": corpus.doc_paths,
        "dev": corpus.valid_queries,
        "eval": corpus.eval_query_paths,
    }
    s3prl_paths = sorted([str(path) for path in s3prl_data[split]])

    dataset = datasets.QUESST14(dataset_root.parent, split)
    torchaudio_paths = sorted([dataset.get_metadata(i)[0] for i in range(len(dataset))])

    assert s3prl_paths, torchaudio_paths
