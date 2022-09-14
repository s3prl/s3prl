from .base import SequentialDataPipe
from .common_pipes import LoadAudio, SetOutputKeys
from .extract_feat_pipes import ExtractOnlineFeat
from .masked_reconstruction_pipes import MaskedReconstruction, PrepareTargetFeat
from .noise_augmentation_pipes import NoiseAugmentation
from .norm_wav_pipes import NormWavDecibel


class PretrainTeraPipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
    """

    def __init__(
        self,
        output_keys: dict = None,
        position_encoding_size: int = 768,
        mask_proportion: float = 0.15,
        mask_consecutive_min: int = 7,
        mask_consecutive_max: int = 7,
        mask_allow_overlap: bool = True,
        mask_bucket_ratio: float = 1.5,
        mask_frequency: int = 0.2,
        noise_proportion: float = 0.0,
        win_ms: int = 25,
        hop_ms: int = 10,
        n_freq: int = 201,
        n_mels: int = 80,
        n_mfcc: int = 13,
        input: dict = {
            "channel": 0,
            "cmvn": True,
            "delta": 0,
            "feat_type": "mel",
            "log": True,
        },
        target: dict = {
            "channel": 1,
            "cmvn": True,
            "delta": 0,
            "feat_type": "mel",
            "log": True,
        },
        target_level: int = -25,
        audio_sample_rate: int = 16000,
        audio_channel_reduction: str = "first",
        n_jobs: int = 6,
    ):
        """
        Args:
            output_keys (dict): args for the output handle
            position_encoding_size (int): this should be identical to `hidden_size`
            mask_proportion (float): mask this percentage of all spectrogram frames in each sequence at random during MAM training
            mask_consecutive_min (int): mask this amount of consecutive frames
            mask_consecutive_max (int): mask this amount of consecutive frames
            mask_allow_overlap (bool): allow overlap masking
            mask_bucket_ratio (float): only used when overlap is not allowed. sample a mask from each bucket in size of [sampled mask_consecutive * mask_bucket_ratio]
            mask_frequency (float): mask maximum this percentage of frequency bands, set to 0 for no frequency mask
            noise_proportion (float): for this percentage of the time, Gaussian noise will be applied on all frames during MAM training, set to 0 for no noise
            win_ms (int): window size in ms
            hop_ms (int): hop size in ms
            n_freq (int): number of frequency bins
            n_mels (int): number of mel features
            n_mfcc (int): number of mfcc features
            input (dict): args for the input feat, example - {"channel": 0, "cmvn": True, "delta": 0, "feat_type": "mel", "log": True,}
            target (dict): args for the output feat, example - {"channel": 1, "cmvn": True, "delta": 0, "feat_type": "mel", "log": True,}
            target_level (int): normalize the wav decibel level to the target value
            audio_sample_rate (int): audio sample rate
            audio_channel_reduction (str): "first" channel
            n_jobs (int): number of workers
        """
        output_keys = output_keys or dict(
            x="masked_feat",
            label="target_feat",
            label_mask="label_mask",
            position_encoding="pos_enc",
            attention_mask="attn_mask",
            unique_name="id",
        )

        super().__init__(
            LoadAudio(
                n_jobs=n_jobs,
                audio_sample_rate=audio_sample_rate,
                audio_channel_reduction=audio_channel_reduction,
            ),
            NormWavDecibel(
                target_level=target_level,
            ),
            ExtractOnlineFeat(
                win_ms=win_ms,
                hop_ms=hop_ms,
                n_freq=n_freq,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                input=input,
                target=target,
                feat_name="source_feat",
            ),
            PrepareTargetFeat(
                use_copy=True,
                source_feat_name="source_feat",
                target_feat_name="target_feat",
            ),
            NoiseAugmentation(
                noise_proportion=noise_proportion,
                input_feat_name="source_feat",
                output_feat_name="noised_feat",
            ),
            MaskedReconstruction(
                position_encoding_size=position_encoding_size,
                mask_proportion=mask_proportion,
                mask_consecutive_min=mask_consecutive_min,
                mask_consecutive_max=mask_consecutive_max,
                mask_allow_overlap=mask_allow_overlap,
                mask_bucket_ratio=mask_bucket_ratio,
                mask_frequency=mask_frequency,
                source_feat_name="noised_feat",
                target_feat_name="target_feat",
                masked_feat_name="masked_feat",
                pos_enc_name="pos_enc",
                attn_mask_name="attn_mask",
                label_mask_name="label_mask",
            ),
            SetOutputKeys(output_keys=output_keys),
        )
