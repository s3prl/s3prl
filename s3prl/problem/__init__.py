"""
Pre-defined python recipes with customizable methods
"""

from .asr.superb_asr import SuperbASR
from .asr.superb_pr import SuperbPR
from .asr.superb_sf import SuperbSF
from .asv.superb_asv import SuperbASV
from .common.hear_beijing_opera import HearBeijingOpera
from .common.hear_cremad import HearCremaD
from .common.hear_dcase_2016_task2 import HearDcase2016Task2
from .common.hear_esc50 import HearESC50
from .common.hear_fsd import HearFSD
from .common.hear_gsc5hr import HearGSC5hr
from .common.hear_gtzan import HearGtzan
from .common.hear_gtzan_music_speech import HearGtzanMusicSpeech
from .common.hear_gunshot import HearGunshot
from .common.hear_libricount import HearLibriCount
from .common.hear_maestro import HearMaestro
from .common.hear_nsynth5hr import HearNsynth5hr
from .common.hear_stroke import HearStroke
from .common.hear_tonic import HearTonic
from .common.hear_vocal import HearVocal
from .common.hear_vox_lingual import HearVoxLingual
from .common.superb_er import SuperbER
from .common.superb_ic import SuperbIC
from .common.superb_ks import SuperbKS
from .common.superb_sid import SuperbSID
from .diarization.superb_sd import SuperbSD

__all__ = [
    "SuperbASR",
    "SuperbPR",
    "SuperbSF",
    "SuperbASV",
    "SuperbER",
    "SuperbIC",
    "SuperbKS",
    "SuperbSID",
    "SuperbSD",
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
