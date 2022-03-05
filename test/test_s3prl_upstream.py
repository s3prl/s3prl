import pytest
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from s3prl.nn.upstream import S3PRLUpstream

device = "cuda" if torch.cuda.is_available() else "cpu"
TOLERATE_FRAME_NUM_DIFF = 3


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", ["apc", "hubert", "wav2vec2", "wavlm", "unispeech_sat", "decoar2"]
)
def test_s3prl_upstream(name):
    upstream = S3PRLUpstream(name).to(device)
    batch_size = 8
    sample_rate = 16000
    seconds = [random.randint(1, 8) for _ in range(batch_size)]
    samples = [sample_rate * sec for sec in seconds]
    wavs = pad_sequence(
        [torch.randn(sample, 1) for sample in samples], batch_first=True
    ).to(device)
    wavs_len = torch.LongTensor(samples).to(device)

    hidden_states, hidden_states_len = upstream(wavs, wavs_len).slice(2)
    for h, hl in zip(hidden_states, hidden_states_len):
        assert abs(h.size(1) - hl.max().item()) < TOLERATE_FRAME_NUM_DIFF
