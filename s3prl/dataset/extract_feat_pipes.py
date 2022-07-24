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
    kaldi: dict = None
    delta: dict = None
    cmvn: dict = None
    wav_name: str = "wav"
    feat_name: str = "feat"
    """
    Args:
        kaldi (dict): args for the kaldi extracter
        delta (dict): args for applying delta on features
        cmvn (dict): args for applying cmvn on features
        wav_name (str): handle for the `takes` (input)
        feat_name (str): handle for the `provides` (output)
    """

    def extract_feat(self, extracter, wav):
        """
        (wav_seq_len, 1) -> extracter -> (feat_seq_len, feat_dim)
        """
        feat = extracter(wav)
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        _audio_config = {
            "kaldi": self.kaldi,
            "delta": self.delta,
            "cmvn": self.cmvn,
        }
        extracter, feat_dim, frame_shift = kaldi_feat_extracter(_audio_config)
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
    win_ms: int = 25
    hop_ms: int = 10
    n_freq: int = 201
    n_mels: int = 80
    n_mfcc: int = 13
    input: dict = None
    target: dict = None
    wav_name: str = "wav"
    feat_name: str = "feat"
    """
    Args:
        win_ms (int): window size in ms
        hop_ms (int): hop size in ms
        n_freq (int): number of frequency bins
        n_mels (int): number of mel features
        n_mfcc (int): number of mfcc features
        input (dict): args for the input feat, example - {"channel": 0, "cmvn": True, "delta": 0, "feat_type": "mel", "log": True,}
        target (dict): args for the output feat, example - {"channel": 1, "cmvn": True, "delta": 0, "feat_type": "mel", "log": True,}
        wav_name (str): handle for the `takes` (input)
        feat_name (str): handle for the `provides` (output)
    """

    def extract_feat(self, extracter, wav):
        """
        (wav_seq_len, 1) -> permute + unsqueeze ->
        (1, 1, wav_seq_len) -> extracter -> (feat_seq_len, feat_dim)
        """
        wav = wav.permute(1, 0).unsqueeze(0)
        feat = extracter(wav)[0][0]
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        _audio_config = {
            "win_ms": self.win_ms,
            "hop_ms": self.hop_ms,
            "n_freq": self.n_freq,
            "n_mels": self.n_mels,
            "n_mfcc": self.n_mfcc,
            "input": self.input,
            "target": self.target,
        }
        extracter, feat_dim, _ = online_feat_extracter(_audio_config)
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
    feat_type: str = "fbank"
    feat_dim: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    decode_wav: bool = False
    cmvn: bool = True
    wav_name: str = "wav"
    feat_name: str = "feat"
    """
    Args:
        feat_type (str): feature type
        feat_dim (int): feature dimension
        frame_length (int): window size in ms
        frame_shift (int): hop size in ms
        decode_wav (bool): whether to decode wav
        cmvn (bool): whether to apply uttr.-wised CMVN on feature
        wav_name (str): handle for the `takes` (input)
        feat_name (str): handle for the `provides` (output)
    """

    def extract_feat(self, extracter, wav):
        """
        (wav_seq_len, 1) -> permute ->
        (1, wav_seq_len) -> extracter -> (feat_seq_len, feat_dim)
        """
        feat = extracter(wav.permute(1, 0))
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        _audio_config = {
            "feat_type": self.feat_type,
            "feat_dim": self.feat_dim,
            "frame_length": self.frame_length,
            "frame_shift": self.frame_shift,
            "decode_wav": self.decode_wav,
            "cmvn": self.cmvn,
        }
        extracter, feat_dim = apc_feat_extracter(_audio_config)
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
    feat_type: str = "fbank"
    feat_dim: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    decode_wav: bool = False
    cmvn: bool = True
    wav_name: str = "wav"
    feat_name: str = "feat"
    """
    Args:
        feat_type (str): feature type
        feat_dim (int): feature dimension
        frame_length (int): window size in ms
        frame_shift (int): hop size in ms
        decode_wav (bool): whether to decode wav
        cmvn (bool): whether to apply uttr.-wised CMVN on feature
        wav_name (str): handle for the `takes` (input)
        feat_name (str): handle for the `provides` (output)
    """

    def extract_feat(self, extracter, wav):
        """
        (wav_seq_len, 1) -> permute ->
        (1, wav_seq_len) -> extracter -> (feat_seq_len, feat_dim)
        """
        feat = extracter(wav.permute(1, 0))
        return feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        _audio_config = {
            "feat_type": self.feat_type,
            "feat_dim": self.feat_dim,
            "frame_length": self.frame_length,
            "frame_shift": self.frame_shift,
            "decode_wav": self.decode_wav,
            "cmvn": self.cmvn,
        }
        extracter, feat_dim = npc_feat_extracter(_audio_config)
        dataset.add_tool("extracter", extracter)
        dataset.add_tool("feat_dim", feat_dim)
        dataset.add_dynamic_item(
            self.extract_feat,
            takes=["extracter", self.wav_name],
            provides=self.feat_name,
        )
        return dataset
