from .base import SequentialDataPipe
from .common_pipes import LoadAudio, SetOutputKeys


class SpeakerClassificationPipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
        label: str
    """

    def __init__(
        self,
        output_keys: dict = None,
        audio_sample_rate: int = 16000,
        audio_channel_reduction: str = "first",
        **kwargs
    ):
        output_keys = output_keys or dict(
            x="wav",
            x_len="wav_len",
            label="label",
            unique_name="id",
        )

        super().__init__(
            LoadAudio(
                audio_sample_rate=audio_sample_rate,
                audio_channel_reduction=audio_channel_reduction,
            ),
            SetOutputKeys(output_keys=output_keys),
        )
