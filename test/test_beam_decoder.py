import torch

from s3prl.nn import BeamDecoder


def test_beam_decoder():
    decoder = BeamDecoder()
    emissions = torch.randn((4, 100, 31))
    emissions = torch.log_softmax(emissions, dim=2)
    hyps = decoder.decode(emissions)
