"""
Parse the commonly used corpus into standardized dictionary structure
"""

from .fluent_speech_commands import FluentSpeechCommands
from .iemocap import IEMOCAP
from .librilight import LibriLight
from .librispeech import LibriSpeech
from .quesst14 import Quesst14
from .snips import SNIPS
from .speech_commands import SpeechCommandsV1
from .voxceleb1sid import VoxCeleb1SID
from .voxceleb1sv import VoxCeleb1SV

__all__ = [
    "FluentSpeechCommands",
    "IEMOCAP",
    "LibriSpeech",
    "LibriLight",
    "Quesst14",
    "SNIPS",
    "SpeechCommandsV1",
    "VoxCeleb1SID",
    "VoxCeleb1SV",
]
