"""
Common model and loss in pure torch.nn.Module with torch dependency only
"""

from .beam_decoder import BeamDecoder
from .common import FrameLevel, UtteranceLevel
from .linear import FrameLevelLinear, MeanPoolingLinear
from .pooling import MeanPooling, TemporalAveragePooling, TemporalStatisticsPooling
from .rnn import RNNEncoder, SuperbDiarizationModel
from .speaker_loss import amsoftmax, softmax
from .speaker_model import SuperbXvector, XVectorBackbone
from .upstream import Featurizer, S3PRLUpstream

__all__ = [
    "S3PRLUpstream",
    "Featurizer",
    "FrameLevel",
    "UtteranceLevel",
    "FrameLevelLinear",
    "MeanPoolingLinear",
    "MeanPooling",
    "TemporalAveragePooling",
    "TemporalStatisticsPooling",
    "RNNEncoder",
    "SuperbDiarizationModel",
    "amsoftmax",
    "softmax",
    "SuperbXvector",
    "XVectorBackbone",
    "BeamDecoder",
]
