import inspect
import logging

from s3prl import Object

logger = logging.getLogger(__name__)


class Child(Object):
    def __init__(self, x, y, *others, a=3, b=4, **kwargs):
        super().__init__()
        assert self.arguments.x == x
        assert self.arguments.y == y
        assert self.arguments.a == a


def test_object():
    child = Child(3, 4)
    child = Child(3, 4, a=5, c=6)
    child = Child(3, 4, 5, 6, a=5, c=6)
