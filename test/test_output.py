import pytest

from s3prl.base.output import Output


def test_output():
    with pytest.raises(ValueError):
        tmp = Output(wavs=3, b=4)

    with pytest.raises(ValueError):
        tmp = Output(wavs_len=3, b=4)

    tmp = Output(wav_len=None, wav=3)
    assert tmp.wav == tmp["wav"]

    tmp.wav = 7  # equals to tmp["wav"] = 7
    tmp[0] = 8  # equals to tmp["wav_len"] = 8
    assert tmp.wav_len == tmp["wav_len"] == 8
    assert tmp.wav == tmp["wav"] == 7
    assert {**tmp} == dict(tmp.items())

    wav, wav_len = tmp.subset("wav", "wav_len")
    wav_len2, wav2 = tmp.slice(2)
    assert wav == wav2 == 7
    assert wav_len == wav_len2 == 8

    wav_len = tmp.slice(1)
    assert wav_len == 8

    wav = tmp.subset("wav")
    assert wav == 7

    del tmp.wav_len
    wav = tmp.slice(1)
    assert wav == 7

    output = Output(output=3, loss=4)
    output.wav = 6
    with pytest.raises(ValueError):
        output.new_result = 5

    assert 3 == output.slice(1)
    assert (3, 4) == output.slice(2)
    assert 4 == output.slice(1, 2)
    assert Output(loss=4) == output.slice(1, 2, as_type="dict")

    assert 3 == output.subset("output")
    assert (4, 3) == output.subset("loss", "output")
    assert (4, 3) == output.subset(1, 0)
    assert 4 == output.subset(1)
    assert Output(loss=4) == output.subset("loss", as_type="dict")
