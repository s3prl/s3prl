import logging

from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)


def wavlm_and_whisper(**kwds):
    return _UpstreamExpert(**kwds)
