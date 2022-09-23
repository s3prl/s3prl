from s3prl.dataset.base import DataPipe, SequentialDataPipe
from s3prl.dataset.chunking import UnfoldChunkBySec
from s3prl.dataset.common_pipes import LoadAudio, SetOutputKeys
from s3prl.dataset.multiclass_tagging import BuildMultiClassTagging


class HearTimestampDatapipe(SequentialDataPipe):
    def __init__(
        self,
        sample_rate: int = 16000,
        feat_frame_shift: int = 160,
    ):
        super().__init__(
            UnfoldChunkBySec(
                min_chunk_secs=4.0,
                max_chunk_secs=4.0,
                step_secs=4.0,
            ),
            LoadAudio(audio_sample_rate=sample_rate),
            BuildMultiClassTagging(
                sample_rate=sample_rate,
                feat_frame_shift=feat_frame_shift,
                intra_or_inter="inter",
                all_category_name="category",
            ),
            SetOutputKeys(
                dict(
                    x="wav",
                    x_len="wav_len",
                    y="multiclass_tag",
                    y_len="tag_len",
                    record_id="unchunked_id",
                    chunk_id="chunk_index",
                )
            ),
        )
