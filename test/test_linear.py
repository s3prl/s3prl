from s3prl.nn import FrameLevelLinear

def test_FrameLevelLinear(helpers):
    module = FrameLevelLinear(3, 4, [5, 6])
    helpers.validate_module(module)
