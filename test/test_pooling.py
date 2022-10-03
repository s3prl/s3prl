import pytest
import torch

from s3prl.nn.common import UtteranceLevel
from s3prl.nn.pooling import (
    AttentiveStatisticsPooling,
    MeanPooling,
    SelfAttentivePooling,
    TemporalStatisticsPooling,
)


@pytest.mark.parametrize(
    "pooling_type",
    [
        "MeanPooling",
        "TemporalStatisticsPooling",
        "AttentiveStatisticsPooling",
        "SelfAttentivePooling",
    ],
)
def test_utterance_level_with_pooling(pooling_type: str):
    model = UtteranceLevel(256, 64, [128], "ReLU", None, pooling_type, None)
    output = model(torch.randn(32, 100, 256), torch.arange(32) + 1)
    assert output.shape == (32, 64)
