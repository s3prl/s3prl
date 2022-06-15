from s3prl.nn.specaug import ModelWithSpecaug


def test_specaug_model():
    model = ModelWithSpecaug(
        input_size=13,
        output_size=25,
        model_cfg=dict(_cls="FrameLevelLinear", hidden_size=32),
    )
    assert model.specaug.apply_time_mask == True
    assert model.specaug.apply_freq_mask == True
