import torch

from s3prl.dataio.dataset.load_audio import LoadAudio
from s3prl.util.pseudo_data import pseudo_audio


def test_load_audio():
    with pseudo_audio([3.0, 4.0, 5.2]) as (paths, num_frames):
        dataset = LoadAudio(paths, [None, 1.0, 3.1], [None, 3.2, None], max_secs=4.2)

        for item in dataset:
            assert isinstance(item["wav"], torch.Tensor)
