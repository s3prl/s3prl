"""
Define how a model is trained & evaluated for each step in the train/valid/test loop
"""

from .base import Task
from .diarization import DiarizationPIT
from .dump_feature import DumpFeature
from .speaker_verification_task import SpeakerVerification
from .speech2text_ctc_task import Speech2TextCTCTask
from .utterance_classification_task import (
    UtteranceClassificationTask,
    UtteranceMultiClassClassificationTask,
)

__all__ = [
    "Task",
    "DiarizationPIT",
    "DumpFeature",
    "SpeakerVerification",
    "Speech2TextCTCTask",
    "UtteranceClassificationTask",
    "UtteranceMultiClassClassificationTask",
]
