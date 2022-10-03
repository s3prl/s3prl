"""
The most common and simple train/valid/test recipes
"""

from .hear_beijing_opera import HearBeijingOpera
from .hear_cremad import HearCremaD
from .hear_dcase_2016_task2 import HearDcase2016Task2
from .hear_esc50 import HearESC50
from .hear_fsd import HearFSD
from .hear_gsc5hr import HearGSC5hr
from .hear_gtzan import HearGtzan
from .hear_gtzan_music_speech import HearGtzanMusicSpeech
from .hear_gunshot import HearGunshot
from .hear_libricount import HearLibriCount
from .hear_maestro import HearMaestro
from .hear_nsynth5hr import HearNsynth5hr
from .hear_stroke import HearStroke
from .hear_tonic import HearTonic
from .hear_vocal import HearVocal
from .hear_vox_lingual import HearVoxLingual
from .superb_er import SuperbER
from .superb_ic import SuperbIC
from .superb_ks import SuperbKS
from .superb_sid import SuperbSID

__all__ = [
    "SuperbER",
    "SuperbIC",
    "SuperbKS",
    "SuperbSID",
    "HearFSD",
    "HearESC50",
    "HearBeijingOpera",
    "HearCremaD",
    "HearGSC5hr",
    "HearGtzanMusicSpeech",
    "HearGtzan",
    "HearGunshot",
    "HearLibriCount",
    "HearNsynth5hr",
    "HearStroke",
    "HearTonic",
    "HearVocal",
    "HearVoxLingual",
    "HearDcase2016Task2",
    "HearMaestro",
]
