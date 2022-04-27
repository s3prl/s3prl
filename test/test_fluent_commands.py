from dotenv import dotenv_values

from s3prl.corpus.fluent_speech_commands import (
    FluentSpeechCommands,
    FluentSpeechCommandsForUtteranceMultiClassClassificataion,
)


def test_fluent_commands():
    config = dotenv_values()
    dataset_root = config["FluentSpeechCommands"]
    dataset = FluentSpeechCommands(dataset_root)
    dataset.data_split_ids
    dataset.data_split
    dataset.all_data

    dataset = FluentSpeechCommandsForUtteranceMultiClassClassificataion(dataset_root)
    train_data, valid_data, test_data, stats = dataset().split(3)
