from dataclasses import dataclass

from .base import AugmentedDynamicItemDataset, DataPipe
from s3prl.upstream.baseline.extracter import get_extracter


@dataclass
class ExtractKaldiFeat(DataPipe):
    audio_config: dict = None
    wav_name: str = "wav"
    feat_name: str = "feat"
    
    def extract_feat(self, extracter, wav):
        return extracter(wav)

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        extracter, output_dim, frame_shift = get_extracter(self.audio_config)
        dataset.add_tool("extracter", extracter)
        dataset.add_tool("output_dim", output_dim)
        dataset.add_tool("frame_shift", frame_shift)
        dataset.add_dynamic_item(self.extract_feat, takes=["extracter", self.wav_name], provides=self.feat_name)
        return dataset
