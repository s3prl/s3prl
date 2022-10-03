from s3prl.nn.linear import FrameLevelLinear
from s3prl.nn.specaug import ModelWithSpecaug


def test_specaug_model():
    model = FrameLevelLinear(input_size=13, output_size=25, hidden_size=32)
    model = ModelWithSpecaug(model)
    assert model.specaug.apply_time_mask == True
    assert model.specaug.apply_freq_mask == True
