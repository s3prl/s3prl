from .base import NNModule
from .linear import FrameLevelLinear, MeanPoolingLinear
from .pooling import MeanPooling
from .rnn import RNNEncoder

from .upstream import S3PRLUpstream
from .upstream_downstream_model import UpstreamDownstreamModel

try:
    from .beam_decoder import BeamDecoder
except ImportError:
    import logging

    logging.warning("Cannot import flashlight, thus cannot use BeamDecoder.")
    BeamDecoder = None

from .speaker_model import speaker_embedding_extractor
from .speaker_loss import amsoftmax, softmax
