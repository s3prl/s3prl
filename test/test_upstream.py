import random

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from wandb import set_trace

from s3prl.nn.upstream import (
    S3PRLUpstream,
    UpstreamDriver,
    UpstreamDownstreamModel,
    get_pseudo_wavs_and_lengths,
)
from s3prl.nn.linear import FrameLevelLinear

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
    for h in hidden_states:
        assert abs(h.size(1) - hidden_states_len.max().item()) < TOLERATE_FRAME_NUM_DIFF


@pytest.mark.slow
def test_upstream_driver():
    layer_selections = [0, 1]
    model = UpstreamDriver(cfg=dict(name="hubert"), layer_selections=layer_selections)
    h, h_len = model(*get_pseudo_wavs_and_lengths()).slice(2)

    model = UpstreamDriver(cfg=dict(name="hubert"))
    h, h_len = model(*get_pseudo_wavs_and_lengths()).slice(2)

    model = UpstreamDriver(cfg=dict(name="hubert"), weighted_sum=False)
    h, h_len = model(*get_pseudo_wavs_and_lengths()).slice(2)


def test_upstream_downstream_model():
    OUTPUT_SIZE = 32
    upstream = UpstreamDriver(cfg=dict(name="fbank"), weighted_sum=False)
    downstream = FrameLevelLinear(upstream.output_size, OUTPUT_SIZE)
    model = UpstreamDownstreamModel(upstream, downstream)
    wavs, wavs_len = get_pseudo_wavs_and_lengths()
    y, y_len = model(wavs, wavs_len).slice(2)

    assert y.size(0) == wavs.size(0)
    assert y.size(-1) == OUTPUT_SIZE
    assert y_len.dim() == 1 and isinstance(y_len, torch.LongTensor)
