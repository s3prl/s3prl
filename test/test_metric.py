from s3prl.metric import cer, per, wer


def isclose(x: float, y: float) -> float:
    return abs(x - y) < 1e-9


def test_metric():
    # test wer & cer
    hyps = ["a ac abb d"]
    refs = ["a ab abc d"]

    assert isclose(cer(hyps, refs), 0.2)
    assert isclose(wer(hyps, refs), 0.5)
    assert isclose(per(hyps, refs), 0.5)
