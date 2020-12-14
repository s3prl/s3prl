import os

import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    unicode_csv_reader,
    walk_files,
)

def load_librispeech_item(fileid, path, ext_audio, ext_txt):

    speaker_id, chapter_id, utterance_id = fileid.split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    return (
        waveform,
        sample_rate,
        utterance
    )

class LIBRISPEECH(Dataset):
    """
    Create a Dataset for LibriSpeech. Each item is a tuple of the form:
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self, path):
        walker = walk_files(
            path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n):
        fileid = self._walker[n]
        return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self):
        return len(self._walker)
