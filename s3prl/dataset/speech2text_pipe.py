from ..encoder.tokenizer import Tokenizer
from .base import SequentialDataPipe
from .common_pipes import (
    EncodeText,
    GenerateTokenizer,
    Grapheme2Phoneme,
    LoadAudio,
    SetOutputKeys,
)


class Speech2TextPipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
        transcription: str
    """

    def __init__(
        self,
        generate_tokenizer: bool = True,
        vocab_type: str = "character",
        text_file: str = None,
        slots_file: str = None,
        vocab_args: dict = None,
        g2p: bool = False,
        labels_name: str = "transcription",
    ):
        output_keys = dict(
            x="wav",
            x_len="wav_len",
            labels=labels_name,
            class_ids="tokenized_text",
            unique_name="id",
        )

        data_pipes = [
            LoadAudio(),
            GenerateTokenizer(
                generate=generate_tokenizer,
                vocab_type=vocab_type,
                text_file=text_file,
                slots_file=slots_file,
                vocab_args=vocab_args,
            ),
        ]

        if not g2p:
            data_pipes.append(EncodeText())
        else:
            data_pipes += [
                Grapheme2Phoneme(),
                EncodeText(text_name="phonemized_text"),
            ]

        data_pipes.append(SetOutputKeys(output_keys=output_keys))

        super().__init__(*data_pipes)
