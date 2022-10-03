import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.fluent_speech_commands import FluentSpeechCommands


@pytest.mark.corpus
def test_fluent_commands():
    config = dotenv_values()
    dataset_root = config["FluentSpeechCommands"]
    dataset = FluentSpeechCommands(dataset_root)
    dataset.data_split_ids
    dataset.data_split
    dataset.all_data
