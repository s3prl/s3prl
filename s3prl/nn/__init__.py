from .base import NNModule
from .common import FrameLevel, UtteranceLevel
from .linear import FrameLevelLinear, MeanPoolingLinear
from .pooling import MeanPooling
from .rnn import RNNEncoder
from .speaker_loss import amsoftmax, softmax
from .speaker_model import SpeakerEmbeddingExtractor
from .upstream import (
    S3PRLUpstream,
    UpstreamDriver,
    S3PRLUpstreamDriver,
    UpstreamDownstreamModel,
)

try:
    from .beam_decoder import BeamDecoder
except ImportError:
    import logging

    logging.warning("Cannot import flashlight, thus cannot use BeamDecoder.")
    BeamDecoder = None
