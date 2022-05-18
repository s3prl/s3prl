from .base import SequentialDataPipe
from .common_pipes import LoadAudio, SetOutputKeys
from .extract_feat_pipes import ExtractOnlineFeat
from .masked_reconstruction_pipes import MaskedReconstruction, PrepareTargetFeat
from .noise_augmentation_pipes import NoiseAugmentation
from .norm_wav_pipes import NormWavDecibel


class PretrainTaskPipe(SequentialDataPipe):
    """
    each item in the input dataset should have:
        wav_path: str
    """

    def __init__(
        self,
        output_keys: dict = None,
        mask_args: dict = None,
        noise_args: dict = None,
        audio_config: dict = None,
        audio_sample_rate: int = 16000,
        audio_channel_reduction: str = "first",
        target_level: int = -25,
        n_jobs: int = 6,
    ):
        output_keys = output_keys or dict(
            x="source_feat",
            label="target_feat",
            label_mask="label_mask",
            position_encoding="pos_enc",
            attention_mask="attn_mask",
            unique_name="id",
        )

        mask_args = mask_args or dict(
            position_encoding_size=768,  # int, this should be identical to `hidden_size`
            mask_proportion=0.15,  # float, mask this percentage of all spectrogram frames in each sequence at random during MAM training
            mask_consecutive_min=7,  # int, mask this amount of consecutive frames
            mask_consecutive_max=7,  # int, mask this amount of consecutive frames
            mask_allow_overlap=True,  # bool, allow overlap masking
            mask_bucket_ratio=1.5,  # float, only used when overlap is not allowed. sample a mask from each bucket in size of [sampled mask_consecutive * mask_bucket_ratio]
            mask_frequency=0,  # int, mask maximum this percentage of frequency bands, set to 0 for no frequency mask
        )

        noise_args = noise_args or dict(
            noise_proportion=0.0,  # float, for this percentage of the time, Gaussian noise will be applied on all frames during MAM training, set to 0 for no noise
        )

        audio_config = audio_config or dict(
            win_ms=25,
            hop_ms=10,
            n_freq=201,
            n_mels=80,
            n_mfcc=13,
            input={
                "channel": 0,
                "cmvn": True,
                "delta": 0,
                "feat_type": "mel",
                "log": True,
            },
            target={
                "channel": 1,
                "cmvn": True,
                "delta": 0,
                "feat_type": "mel",
                "log": True,
            },
        )

        super().__init__(
            LoadAudio(
                n_jobs=n_jobs,
                audio_sample_rate=audio_sample_rate,
                audio_channel_reduction=audio_channel_reduction,
            ),
            NormWavDecibel(target_level=target_level),
            ExtractOnlineFeat(audio_config=audio_config, feat_name="source_feat"),
            PrepareTargetFeat(
                use_copy=True,
                source_feat_name="source_feat",
                target_feat_name="target_feat",
            ),
            NoiseAugmentation(
                noise_args=noise_args,
                input_feat_name="source_feat",
                output_feat_name="source_feat",
            ),
            MaskedReconstruction(
                mask_args=mask_args,
                source_feat_name="source_feat",
                target_feat_name="target_feat",
                pos_enc_name="pos_enc",
                attn_mask_name="attn_mask",
                label_mask_name="label_mask",
            ),
            SetOutputKeys(output_keys=output_keys),
        )
