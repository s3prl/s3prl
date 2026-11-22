from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import yaml

from s3prl.downstream.timit_phone.model import ConvBank


@pytest.fixture
def default_model():
    config_path = (
        Path(__file__).resolve().parents[1] / "s3prl/downstream/timit_phone/config.yaml"
    )
    with config_path.open(encoding="utf-8") as config_file:
        model_config = yaml.safe_load(config_file)["downstream_expert"]["modelrc"]
    assert model_config["select"] == "ConvBank"
    torch.manual_seed(0)
    model = ConvBank(768, 39, **model_config["ConvBank"])
    return model, torch.randn(2, 50, 768)


def dropout_free_logits(model, features):
    hidden = model.in_linear(features).relu().transpose(1, 2).contiguous()
    responses = [conv(hidden) for conv in model.cnns]
    hidden = torch.cat(responses, dim=1).transpose(1, 2).contiguous()
    return model.out_linear(hidden.relu())


def test_default_configuration_shape_and_finite_logits(default_model):
    model, features = default_model
    assert model.drop_p == 0.5
    assert features.shape == (2, 50, 768)
    with torch.no_grad():
        logits = model.eval()(features)
    assert logits.shape == (2, 50, 39)
    assert torch.isfinite(logits).all()


def test_eval_logits_are_repeatable(default_model):
    model, features = default_model
    model.eval()
    with torch.no_grad():
        first = model(features)
        second = model(features)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_eval_matches_dropout_free_reference(default_model):
    model, features = default_model
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(features), dropout_free_logits(model, features), rtol=0, atol=0
        )


def test_both_dropout_sites_follow_mode_transitions(default_model):
    model, features = default_model
    original_dropout = F.dropout
    calls = []

    def record_dropout(inputs, p=0.5, training=True, inplace=False):
        outputs = original_dropout(inputs, p, training, inplace)
        masked = bool(((inputs != 0) & (outputs == 0)).any())
        calls.append((p, training, masked))
        return outputs

    with torch.no_grad(), patch.object(F, "dropout", side_effect=record_dropout):
        for training in (True, False, True, False):
            model.train(training)
            calls.clear()
            model(features)
            assert calls == [(0.5, training, training)] * 2


def test_training_backward_and_optimizer_step(default_model):
    model, features = default_model
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logits = model(features)
    targets = torch.randint(logits.shape[-1], logits.shape[:-1])
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    assert torch.isfinite(loss)
    loss.backward()
    for layer in (model.in_linear, *model.cnns, model.out_linear):
        gradient = layer.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)
    previous_weights = model.out_linear.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(previous_weights, model.out_linear.weight)


def test_default_checkpoint_parameter_layout(default_model):
    model, _ = default_model
    expected_shapes = {
        "in_linear.weight": (64, 768),
        "in_linear.bias": (64,),
        "cnns.0.weight": (32, 64, 3),
        "cnns.0.bias": (32,),
        "cnns.1.weight": (32, 64, 5),
        "cnns.1.bias": (32,),
        "cnns.2.weight": (32, 64, 7),
        "cnns.2.bias": (32,),
        "out_linear.weight": (39, 96),
        "out_linear.bias": (39,),
    }
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == (
        expected_shapes
    )


@pytest.mark.parametrize(
    "batch, frames, kernels, classes", [(1, 7, [3], 3), (3, 19, [1, 5], 11)]
)
def test_zero_dropout_and_output_shape(batch, frames, kernels, classes):
    torch.manual_seed(0)
    model = ConvBank(16, classes, kernels, cnn_size=4, hidden_size=8, dropout=0)
    features = torch.randn(batch, frames, 16)
    with torch.no_grad():
        expected = dropout_free_logits(model, features)
        assert expected.shape == (batch, frames, classes)
        torch.testing.assert_close(model.train()(features), expected, rtol=0, atol=0)
        torch.testing.assert_close(model.eval()(features), expected, rtol=0, atol=0)
