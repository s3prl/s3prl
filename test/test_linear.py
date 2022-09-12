import torch

from s3prl.nn import FrameLevel


def test_FrameLevel(helpers):
    module = FrameLevel(3, 4, [5, 6])
    x = torch.randn(32, 10, 3)
    x_len = (torch.ones(32) * 3).long()
    h, hl = module(x, x_len)
