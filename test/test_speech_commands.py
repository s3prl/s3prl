from dotenv import dotenv_values

from s3prl.corpus.speech_commands import SpeechCommandsV1ForSUPERB


def test_speech_commands():
    config = dotenv_values()
    corpus = SpeechCommandsV1ForSUPERB(config["SpeechCommandsV1"])
    train, valid, test = corpus().slice(3)
    corpus.all_data
