import pytest
from dotenv import dotenv_values

from s3prl.corpus.speech_commands import SpeechCommandsV1


@pytest.mark.corpus
def test_speech_commands():
    env = dotenv_values()
    corpus = SpeechCommandsV1(env["GSC1"])
    all_data = corpus.all_data
    classes = set([value["class_name"] for key, value in all_data.items()])
    assert len(classes) == 12, f"{classes}"
