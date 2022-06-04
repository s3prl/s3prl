from s3prl import Container
from s3prl.util import registry


@registry.put("new_class")
class NewClass:
    def __init__(self) -> None:
        pass


@registry.put()
def example_func():
    pass


def test_registry():
    assert registry.get("new_class") == NewClass
    assert registry.get("example_func") == example_func


def test_container_with_registry():
    config = Container(stage_1=dict(_cls="new_class"))
    assert config.stage_1._cls == NewClass
