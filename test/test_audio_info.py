from pathlib import Path

from s3prl.util.audio_info import get_audio_info
from s3prl.util.pseudo_data import pseudo_audio


def test_audio_info():
    with pseudo_audio([3.0, 4.1, 1.1]) as (paths, num_samples):
        infos = get_audio_info(paths, [Path(path).stem for path in paths])

    assert infos[0]["num_frames"] == 3 * 16000
