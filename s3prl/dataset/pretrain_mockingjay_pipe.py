from .base import SequentialDataPipe
from .common_pipes import LoadAudio, SetOutputKeys
from .extract_feat_pipes import ExtractKaldiFeat
from .masked_reconstruction_pipes import MaskedReconstruction, PrepareTargetFeat


class PretrainMockingjayPipe(SequentialDataPipe):
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
        kaldi: dict = {
            "feat_type": "fbank",
            "fbank": {
                "frame_length": 25.0,
                "frame_shift": 10.0,
                "num_mel_bins": 80,  # because delta={"order": 2}
                "use_log_fbank": True,
            },
            "mfcc": {"frame_length": 25.0, "frame_shift": 10.0, "num_ceps": 13},
            "spectrogram": {"frame_length": 25.0, "frame_shift": 10.0},
        },
        delta: dict = {"order": 2, "win_length": 5},
        cmvn: dict = {"use_cmvn": True},
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
            kaldi (dict): args for the kaldi extracter
            delta (dict): args for applying delta on features
            cmvn (dict): args for applying cmvn on features
            n_mels (int): number of mel features
            n_mfcc (int): number of mfcc features
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
            ExtractKaldiFeat(
                kaldi=kaldi, delta=delta, cmvn=cmvn, feat_name="source_feat"
            ),
            PrepareTargetFeat(
                use_copy=True,
                source_feat_name="source_feat",
                target_feat_name="target_feat",
            ),
            MaskedReconstruction(
                position_encoding_size=position_encoding_size,
                mask_proportion=mask_proportion,
                mask_consecutive_min=mask_consecutive_min,
                mask_consecutive_max=mask_consecutive_max,
                mask_allow_overlap=mask_allow_overlap,
                mask_bucket_ratio=mask_bucket_ratio,
                mask_frequency=mask_frequency,
                source_feat_name="source_feat",
                target_feat_name="target_feat",
                masked_feat_name="masked_feat",
                pos_enc_name="pos_enc",
                attn_mask_name="attn_mask",
                label_mask_name="label_mask",
            ),
            SetOutputKeys(output_keys=output_keys),
        )
