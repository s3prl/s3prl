from dotenv import dotenv_values

from s3prl.corpus.librispeech import LibriSpeech, LibriSpeechForSUPERB


libri_stats = {
    "train-clean-100": 28539,
    "train-clean-360": 104014,
    "train-other-500": 148688,
    "dev-clean": 2703,
    "dev-other": 2864,
    "test-clean": 2620,
    "test-other": 2939,
}


def test_librispeech():
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

    for key, val in libri_stats.items():
        assert len(dataset.get_corpus_splits([key])) == val

    corpus = LibriSpeechForSUPERB(dataset_root)
    train_data, valid_data, test_data = corpus().slice(3)

    assert len(train_data) == libri_stats["train-clean-100"]
    assert len(valid_data) == libri_stats["dev-clean"]
    assert len(test_data) == libri_stats["test-clean"]
