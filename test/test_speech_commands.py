from collections import Counter

import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.speech_commands import SpeechCommandsV1


def _class_counter(data_dict):
    counter = Counter()
    for data_id, data in data_dict.items():
        counter.update([data["class_name"]])
    return counter


@pytest.mark.corpus
def test_speech_commands():
    env = dotenv_values()
    corpus = SpeechCommandsV1(env["GSC1"], env["GSC1_TEST"])
    all_data = corpus.all_data
    classes = set([value["class_name"] for key, value in all_data.items()])
    assert len(classes) == 12, f"{classes}"

    train, valid, test = corpus.data_split
    train_class_counter = _class_counter(train)
    valid_class_counter = _class_counter(valid)
    test_class_counter = _class_counter(test)

    # These pre-defined numbers are obtained with the old DownstreamExpert
    assert train_class_counter == Counter(
        {
            "_unknown_": 32550,
            "stop": 1885,
            "on": 1864,
            "go": 1861,
            "yes": 1860,
            "no": 1853,
            "right": 1852,
            "up": 1843,
            "down": 1842,
            "left": 1839,
            "off": 1839,
            "_silence_": 6,
        }
    )
    assert valid_class_counter == Counter(
        {
            "_unknown_": 4221,
            "stop": 246,
            "on": 257,
            "go": 260,
            "yes": 261,
            "no": 270,
            "right": 256,
            "up": 260,
            "down": 264,
            "left": 247,
            "off": 256,
            "_silence_": 6,
        }
    )
    assert test_class_counter == Counter(
        {
            "_unknown_": 257,
            "stop": 249,
            "on": 246,
            "go": 251,
            "yes": 256,
            "no": 252,
            "right": 259,
            "up": 272,
            "down": 253,
            "left": 267,
            "off": 262,
            "_silence_": 257,
        }
    )
