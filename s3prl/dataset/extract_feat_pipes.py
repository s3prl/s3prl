from dataclasses import dataclass

from s3prl.upstream.baseline.extracter import get_extracter
from s3prl.upstream.baseline.preprocessor import get_preprocessor

from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class ExtractKaldiFeat(DataPipe):
    audio_config: dict = None
    wav_name: str = "wav"
    feat_name: str = "feat"

    def extract_feat(self, extracter, wav):
        feat = extracter(wav)
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        extracter, output_dim, frame_shift = get_extracter(self.audio_config)
        dataset.add_tool("extracter", extracter)
        dataset.add_tool("output_dim", output_dim)
        dataset.add_tool("frame_shift", frame_shift)
        dataset.add_dynamic_item(
            self.extract_feat,
            takes=["extracter", self.wav_name],
            provides=self.feat_name,
        )
        return dataset


@dataclass
class ExtractOnlineFeat(DataPipe):
    audio_config: dict = None
    wav_name: str = "wav"
    feat_name: str = "feat"

    def extract_feat(self, extracter, wav):
        wav = wav.permute(1, 0).unsqueeze(0)  # (seq_len, 1) -> (1, 1, seq_len)
        feat = extracter(wav)[0][0]
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        extracter, output_dim, _ = get_preprocessor(self.audio_config)
        dataset.add_tool("extracter", extracter)
        dataset.add_tool("output_dim", output_dim)
        dataset.add_dynamic_item(
            self.extract_feat,
            takes=["extracter", self.wav_name],
            provides=self.feat_name,
        )
        return dataset
