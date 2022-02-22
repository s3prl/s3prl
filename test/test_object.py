import pytest

from s3prl import Object, init


class Child(Object):
    @init.method
    def __init__(self, x, y):
        super().__init__()


def test_object():
    class FirstChild(Object):
        @init.method
        def __init__(self):
            super().__init__()

    with pytest.raises(AssertionError):

        class SecondChild(Object):
            def __init__(self):
                super().__init__()

    with pytest.raises(AssertionError):

        class FirstChildChild(FirstChild):
            def __init__(self):
                super().__init__()

    child = Child(3, 4)
    assert child.arguments.x == 3
    assert child.arguments.y == 4
