from s3prl import Container, newdict


def test_newdict():
    a = dict(a=3, b=4, c=dict(x=4))
    a = Container(a)
    assert isinstance(a.c, Container)

    b = dict(a=5, c=dict(x=8))
    a.override(b)
    assert a.c.x == 8
    assert a.a == 5

    b = dict(a=dict(k=5), c=newdict(k=4))
    a.override(b)
    assert a.a == dict(k=5)
    assert "x" not in a.c
    assert a.c.k == 4
