from s3prl import Container
from s3prl.util.configuration import parse_override


def test_configuration():
    config = Container(parse_override(
        "optimizer.lr=1.0e-3,,optimizer.name='AdamW',,runner.eval_dataloaders=['dev', 'test']"
    ))
    assert config.optimizer.lr == 1.0e-3
    assert config.optimizer.name == "AdamW"
    assert config.runner.eval_dataloaders == ['dev', 'test']
