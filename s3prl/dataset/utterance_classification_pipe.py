from .base import SequentialDataPipe
from .common_pipes import (
    EncodeCategory,
    EncodeMultiLabel,
    EncodeMultipleCategory,
    LoadAudio,
    SetOutputKeys,
)


class UtteranceClassificationPipe(SequentialDataPipe):
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
        sox_effects: list = None,
        train_category_encoder: bool = False,
    ):
        output_keys = output_keys or dict(
            x="wav",
            x_len="wav_len",
            class_id="class_id",
            label="label",
            unique_name="id",
        )

        super().__init__(
            LoadAudio(
                audio_sample_rate=audio_sample_rate,
                audio_channel_reduction=audio_channel_reduction,
                sox_effects=sox_effects,
            ),
            EncodeCategory(train_category_encoder=train_category_encoder),
            SetOutputKeys(output_keys=output_keys),
        )


class UtteranceMultipleCategoryClassificationPipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
        labels: List[str]
    """

    def __init__(
        self,
        output_keys: dict = None,
        audio_sample_rate: int = 16000,
        audio_channel_reduction: str = "first",
        sox_effects: list = None,
        train_category_encoder: bool = False,
    ):
        output_keys = output_keys or dict(
            x="wav",
            x_len="wav_len",
            class_ids="class_ids",
            labels="labels",
            unique_name="id",
        )

        super().__init__(
            LoadAudio(
                audio_sample_rate=audio_sample_rate,
                audio_channel_reduction=audio_channel_reduction,
                sox_effects=sox_effects,
            ),
            EncodeMultipleCategory(train_category_encoder=train_category_encoder),
            SetOutputKeys(output_keys=output_keys),
        )


class HearScenePipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
        labels: List[str]
    """

    def __init__(
        self,
        output_keys: dict = None,
        audio_sample_rate: int = 16000,
        audio_channel_reduction: str = "first",
    ):
        output_keys = output_keys or dict(
            x="wav",
            x_len="wav_len",
            y="binary_labels",
            labels="labels",
            unique_name="id",
        )

        super().__init__(
            LoadAudio(
                audio_sample_rate=audio_sample_rate,
                audio_channel_reduction=audio_channel_reduction,
            ),
            EncodeMultiLabel(),
            SetOutputKeys(output_keys=output_keys),
        )
