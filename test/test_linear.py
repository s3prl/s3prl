import torch

from s3prl.nn import FrameLevel


def test_FrameLevel(helpers):
    module = FrameLevel(3, 4, [5, 6])
    helpers.validate_module(module, torch.randn(32, module.input_size))
