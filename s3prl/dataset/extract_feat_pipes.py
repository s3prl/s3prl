from dataclasses import dataclass

from s3prl.upstream.apc.audio import create_transform as apc_feat_extracter
from s3prl.upstream.baseline.extracter import get_extracter as kaldi_feat_extracter
from s3prl.upstream.baseline.preprocessor import (
    get_preprocessor as online_feat_extracter,
)
from s3prl.upstream.npc.audio import create_transform as npc_feat_extracter

from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class ExtractKaldiFeat(DataPipe):
    audio_config: dict = None
    wav_name: str = "wav"
    feat_name: str = "feat"

    def extract_feat(self, extracter, wav):
        """
        (wav_seq_len, 1) -> extracter -> (feat_seq_len, feat_dim)
        """
        feat = extracter(wav)
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        extracter, feat_dim, frame_shift = kaldi_feat_extracter(self.audio_config)
        dataset.add_tool("extracter", extracter)
        dataset.add_tool("feat_dim", feat_dim)
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
        """
        (wav_seq_len, 1) -> permute + unsqueeze ->
        (1, 1, wav_seq_len) -> extracter -> (feat_seq_len, feat_dim)
        """
        wav = wav.permute(1, 0).unsqueeze(0)
        feat = extracter(wav)[0][0]
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        extracter, feat_dim, _ = online_feat_extracter(self.audio_config)
        dataset.add_tool("extracter", extracter)
        dataset.add_tool("feat_dim", feat_dim)
        dataset.add_dynamic_item(
            self.extract_feat,
            takes=["extracter", self.wav_name],
            provides=self.feat_name,
        )
        return dataset


@dataclass
class ExtractApcFeat(DataPipe):
    audio_config: dict = None
    wav_name: str = "wav"
    feat_name: str = "feat"

    def extract_feat(self, extracter, wav):
        """
        (wav_seq_len, 1) -> permute ->
        (1, wav_seq_len) -> extracter -> (feat_seq_len, feat_dim)
        """
        feat = extracter(wav.permute(1, 0))
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        extracter, feat_dim = apc_feat_extracter(self.audio_config)
        dataset.add_tool("extracter", extracter)
        dataset.add_tool("feat_dim", feat_dim)
        dataset.add_dynamic_item(
            self.extract_feat,
            takes=["extracter", self.wav_name],
            provides=self.feat_name,
        )
        return dataset


@dataclass
class ExtractNpcFeat(DataPipe):
    audio_config: dict = None
    wav_name: str = "wav"
    feat_name: str = "feat"

    def extract_feat(self, extracter, wav):
        """
        (wav_seq_len, 1) -> permute ->
        (1, wav_seq_len) -> extracter -> (feat_seq_len, feat_dim)
        """
        feat = extracter(wav.permute(1, 0))
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        extracter, feat_dim = npc_feat_extracter(self.audio_config)
        dataset.add_tool("extracter", extracter)
        dataset.add_tool("feat_dim", feat_dim)
        dataset.add_dynamic_item(
            self.extract_feat,
            takes=["extracter", self.wav_name],
            provides=self.feat_name,
        )
        return dataset
