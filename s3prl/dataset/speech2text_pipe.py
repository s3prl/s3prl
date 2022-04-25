from .base import SequentialDataPipe
from .common_pipes import (
    SetOutputKeys,
    LoadAudio,
    EncodeText,
)
from ..encoder.tokenizer import Tokenizer


class Speech2TextPipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
        transcription: str
    """

    def __init__(
        self,
        train_vocab: bool = False,
        tokenizer: Tokenizer = None,
        vocab_type: str = "character",
        text_file: str = None,
        slots_file: str = None,
        vocab_args: dict = None,
    ):
        output_keys = dict(
            x="wav",
            x_len="wav_len",
            labels="transcription",
            class_ids="tokenized_text",
            unique_name="id",
        )

        super().__init__(
            LoadAudio(),
            EncodeText(
                tokenizer=tokenizer,
                train_vocab=train_vocab,
                vocab_type=vocab_type,
                text_file=text_file,
                slots_file=slots_file,
                vocab_args=vocab_args,
            ),
            SetOutputKeys(output_keys=output_keys),
        )
