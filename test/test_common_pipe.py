from collections import OrderedDict

from s3prl.dataset.common_pipes import LoadAudio
from s3prl.util.pseudo_data import pseudo_audio


def test_load_audio():
    with pseudo_audio([8, 20, 15], 16000) as (audio_files, num_samples):
        data = OrderedDict(
            {i: dict(wav_path=file) for i, file in enumerate(audio_files)}
        )
        dataset = LoadAudio()(data)
        assert len(dataset[0]["wav"]) == num_samples[0]

        data[0]["start_sec"] = 1
        data[0]["end_sec"] = 3
        dataset = LoadAudio()(OrderedDict({0: data[0]}))
        assert len(dataset[0]["wav"]) == 32000
