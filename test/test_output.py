import pytest

from s3prl.base.output import Output

def test_output():
    with pytest.raises(AssertionError):
        tmp = Output(n=3, b=4)
    tmp = Output(wav=3, wav_len=4)