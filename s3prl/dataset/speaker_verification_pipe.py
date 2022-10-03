from typing import List

from .base import SequentialDataPipe
from .common_pipes import LoadAudio, RandomCrop, SetOutputKeys


class SpeakerVerificationPipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
        label: str
    """

    def __init__(
        self,
        audio_sample_rate: int = 16000,
        audio_channel_reduction: str = "first",
        random_crop_secs: float = -1,
        sox_effects: List[List] = None,
    ):
        pipes = [
            LoadAudio(
                audio_sample_rate=audio_sample_rate,
                audio_channel_reduction=audio_channel_reduction,
                sox_effects=sox_effects,
            ),
        ]
        output_keys = dict(
            x="wav",
            x_len="wav_len",
            label="label",
            unique_name="id",
        )

        if random_crop_secs != -1:
            pipes.append(
                RandomCrop(sample_rate=audio_sample_rate, max_secs=random_crop_secs)
            )
            output_keys["x"] = "wav_crop"
            output_keys["x_len"] = "wav_crop_len"

        pipes.append(SetOutputKeys(output_keys))
        super().__init__(*pipes)
