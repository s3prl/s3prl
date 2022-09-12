import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.librilight import LibriLight
from s3prl.dataio.corpus.librispeech import LibriSpeech

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
        train_split=[
            "train-clean-100",
            "train-clean-360",
        ],  # FIXME (Leo): I temporary do not have space for train-other-500 ...
        valid_split=["dev-clean", "dev-other"],
        test_split=["test-clean", "test-other"],
    )
    data = dataset.all_data
    assert len(data) == 292367 - libri_stats["train-other-500"]


@pytest.mark.corpus
def test_librilight():
    config = dotenv_values()
    train_corpus = LibriLight(config["LibriLight"])
    eval_corpus = LibriSpeech(config["LibriSpeech"], 4, [])

    train_data = train_corpus.all_data
    _, valid_data, test_data = eval_corpus.data_split

    assert len(train_data) == 48
