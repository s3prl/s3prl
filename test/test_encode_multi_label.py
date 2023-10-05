from collections import OrderedDict

from s3prl.dataset.common_pipes import EncodeMultiLabel
from s3prl.dataset.utterance_classification_pipe import HearScenePipe
from s3prl.util.pseudo_data import pseudo_audio


def test_multi_label():
    data = OrderedDict(
        {
            "0": dict(labels=["hello", "best", "because", "maybe"]),
            "1": dict(labels=["hello", "because", "cat"]),
            "2": dict(labels=["cat", "dog"]),
        }
    )
    dataset = EncodeMultiLabel()(data)
    category = dataset.get_tool("category")
    assert "binary_labels" in dataset[0]


def test_hear_scene():
    with pseudo_audio([8, 10, 3], sample_rate=16000) as (paths, lengths):
        data = OrderedDict(
            {
                "0": dict(
                    wav_path=paths[0], labels=["hello", "best", "because", "maybe"]
                ),
                "1": dict(wav_path=paths[1], labels=["hello", "because", "cat"]),
                "2": dict(wav_path=paths[2], labels=["cat", "dog"]),
            }
        )
        dataset = HearScenePipe()(data)
        item = dataset[0]

        assert "x" in item
        assert "x_len" in item
        assert "y" in item
        assert "labels" in item
        assert "unique_name" in item
