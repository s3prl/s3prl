import pytest
from dotenv import dotenv_values

from s3prl.corpus.librilight import librilight_for_speech2text
from s3prl.corpus.librispeech import LibriSpeech, librispeech_for_speech2text

libri_stats = {
    "train-clean-100": 28539,
    "train-clean-360": 104014,
    "train-other-500": 148688,
    "dev-clean": 2703,
    "dev-other": 2864,
    "test-clean": 2620,
    "test-other": 2939,
}


@pytest.mark.corpus
def test_librispeech_dataset():
    config = dotenv_values()
    dataset_root = config["LibriSpeech"]
    dataset = LibriSpeech(
        dataset_root,
        train_split=["train-clean-100", "train-clean-360", "train-other-500"],
        valid_split=["dev-clean", "dev-other"],
        test_split=["test-clean", "test-other"],
    )
    data = dataset.all_data
    assert len(data) == 292367


@pytest.mark.corpus
def test_librispeech_parsing():
    config = dotenv_values()
    train_data, valid_data, test_data = librispeech_for_speech2text(
        config["LibriSpeech"]
    ).values()

    assert len(train_data) == libri_stats["train-clean-100"]
    assert len(valid_data) == libri_stats["dev-clean"]
    assert len(test_data) == libri_stats["test-clean"]

    keys = list(train_data.keys())
    assert keys[0] == "103-1240-0000"


@pytest.mark.corpus
def test_librilight():
    config = dotenv_values()
    train_data, valid_data, test_data = librilight_for_speech2text(
        config["LibriLight"], config["LibriSpeech"]
    )
    assert len(train_data) == 48
