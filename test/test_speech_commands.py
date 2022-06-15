import pytest
from dotenv import dotenv_values

from s3prl.corpus.speech_commands import gsc_v1_for_superb


@pytest.mark.corpus
def test_speech_commands():
    config = dotenv_values()
    train, valid, test = gsc_v1_for_superb(config["SpeechCommandsV1"]).slice(3)
