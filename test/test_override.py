from s3prl import Container
from s3prl.util.override import parse_overrides


def test_parse_overrides():
    a = Container(a=3, b=dict(k=4))
    x = a.clone().override(parse_overrides("--a 4 --b newdict(c=5)".split(" ")))
    assert x.b.c == 5
    assert "k" not in x.b
    y = a.clone().override(parse_overrides("--a 4 --b dict(c=5)".split(" ")))
    assert y.b.c == 5
    assert y.b.k == 4
