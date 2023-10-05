from .autoregressive_prediction_pipes import AutoregressivePrediction
from .base import SequentialDataPipe
from .common_pipes import LoadAudio, SetOutputKeys
from .extract_feat_pipes import ExtractApcFeat


class PretrainApcPipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
    """

    def __init__(
        self,
        output_keys: dict = None,
        n_future: int = 5,
        feat_type: str = "fbank",
        feat_dim: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        decode_wav: bool = False,
        cmvn: bool = True,
        audio_sample_rate: int = 16000,
        audio_channel_reduction: str = "first",
        n_jobs: int = 6,
    ):
        """
        Args:
            output_keys (dict): args for the output handle
            n_future (int): number of future steps for the autoregressive predictive task
            feat_type (str): feature type
            feat_dim (int): feature dimension
            frame_length (int): window size in ms
            frame_shift (int): hop size in ms
            decode_wav (bool): whether to decode wav
            cmvn (bool): whether to apply uttr.-wised CMVN on feature
            audio_sample_rate (int): audio sample rate
            audio_channel_reduction (str): "first" channel
            n_jobs (int): number of workers
        """
        output_keys = output_keys or dict(
            x="source_feat",
            label="target_feat",
            x_len="feat_len",
            unique_name="id",
        )

        super().__init__(
            LoadAudio(
                n_jobs=n_jobs,
                audio_sample_rate=audio_sample_rate,
                audio_channel_reduction=audio_channel_reduction,
            ),
            ExtractApcFeat(
                feat_type=feat_type,
                feat_dim=feat_dim,
                frame_length=frame_length,
                frame_shift=frame_shift,
                decode_wav=decode_wav,
                cmvn=cmvn,
                feat_name="source_feat",
            ),
            AutoregressivePrediction(
                n_future=n_future,
                source_feat_name="source_feat",
                target_feat_name="target_feat",
                source_feat_len_name="feat_len",
            ),
            SetOutputKeys(output_keys=output_keys),
        )
