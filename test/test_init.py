from argparse import Namespace
from multiprocessing.context import set_spawning_popen

import pytest

from s3prl import init


@init.method
def create_model(input_size, output_size, *layers, trainable=False, **kwargs):
    return Namespace(
        **dict(
            input_size=input_size,
            output_size=output_size,
            layers=layers,
            trainable=trainable,
            kwargs=kwargs,
        )
    )


@pytest.mark.parametrize("input_size", [3, 4])
@pytest.mark.parametrize("output_size", [4, 5])
@pytest.mark.parametrize("layers", [("a", "b"), ("c", "d", "e")])
@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize(
    "kwargs", [dict(), dict(device="cpu"), dict(testing=False, upstream="hubert")]
)
def test_create_model(input_size, output_size, layers, trainable, kwargs):
    model = create_model(3, 4, *layers, trainable=trainable)
    composite_model = create_model(input_size, output_size, model, **kwargs)

    serialized = init.serialize(composite_model)
    new_composite_model = init.deserialize(serialized)

    assert composite_model == new_composite_model
