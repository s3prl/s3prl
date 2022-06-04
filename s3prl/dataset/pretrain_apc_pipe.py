from .autoregressive_prediction_pipes import AutoregressivePrediction
from .base import SequentialDataPipe
from .common_pipes import LoadAudio, SetOutputKeys
from .extract_feat_pipes import ExtractApcFeat


class PretrainTaskPipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
    """

    def __init__(
        self,
        output_keys: dict = None,
        n_future: int = 5,
        audio_config: dict = None,
        audio_sample_rate: int = 16000,
        audio_channel_reduction: str = "first",
        n_jobs: int = 6,
    ):
        output_keys = output_keys or dict(
            x="source_feat",
            label="target_feat",
            x_len="feat_len",
            unique_name="id",
        )

        audio_config = audio_config or dict(
            feat_type="fbank",  # Feature type
            feat_dim=80,  # Feature dimension
            frame_length=25,  # Window size in ms
            frame_shift=10,  # Hop size in ms
            decode_wav=False,
            cmvn=True,  # Apply uttr.-wised CMVN on Mel spectrogram
        )

        super().__init__(
            LoadAudio(
                n_jobs=n_jobs,
                audio_sample_rate=audio_sample_rate,
                audio_channel_reduction=audio_channel_reduction,
            ),
            ExtractApcFeat(audio_config=audio_config, feat_name="source_feat"),
            AutoregressivePrediction(
                n_future=n_future,
                source_feat_name="source_feat",
                target_feat_name="target_feat",
                source_feat_len_name="feat_len",
            ),
            SetOutputKeys(output_keys=output_keys),
        )
