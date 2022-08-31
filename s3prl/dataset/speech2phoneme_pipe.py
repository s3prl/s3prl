from .base import SequentialDataPipe
from .common_pipes import LoadAudio, Phonemize, SetOutputKeys


class Speech2PhonemePipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
        transcription: str
    """

    def __init__(self):
        output_keys = dict(
            x="wav",
            x_len="wav_len",
            labels="phonemized_text",
            class_ids="tokenized_text",
            unique_name="id",
        )

        super().__init__(
            LoadAudio(),
            Phonemize(),
            SetOutputKeys(output_keys=output_keys),
        )
