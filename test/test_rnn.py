import torch

from s3prl.nn import RNNEncoder


def test_rnn(helpers):
    modules = [
        RNNEncoder(
            input_size=8,
            output_size=6,
            module="LSTM",
            hidden_size=[10, 10, 10],
            dropout=[0.1, 0.1, 0.1],
            layer_norm=[True, True, True],
            proj=[True, True, True],
            sample_rate=[1, 2, 1],
            sample_style="drop",
            bidirectional=True,
        ),
        RNNEncoder(
            input_size=8,
            output_size=6,
            module="LSTM",
            hidden_size=[10, 10, 10],
            dropout=[0.1, 0.1, 0.1],
            layer_norm=[True, True, True],
            proj=[True, True, True],
            sample_rate=[1, 2, 1],
            sample_style="concat",
            bidirectional=True,
        ),
    ]

    for module in modules:
        xs = torch.randn(32, 50, module.input_size)
        xs_len = torch.arange(32) + (50 - 32) + 1

        out, out_len = module(xs, xs_len)
        assert out.shape[1] == 25
        assert out.shape[2] == module.output_size
        assert out_len.max() == 25
