import pytest

from s3prl.base.container import Container, field


def test_container_to_dict():
    config = Container(
        a=3,
        b=Container(
            c=4,
        ),
    )
    config_dict = dict(
        a=3,
        b=dict(c=4),
    )
    assert config == config_dict


def test_container_unfilled_fields():
    config = Container(a="hello", b=3, c=dict(x=dict(y="???")), z=["he", "???", "best"])
    assert config.list_unfilled_fields() == ["c.x.y", "z.[1]"]

    config = Container(a=4, b=5)
    assert len(config.unfilled_fields()) == 0


def test_container_cls():
    from s3prl.nn.pooling import MeanPooling

    config = Container(_cls="s3prl.nn.pooling.MeanPooling")
    assert config._cls == MeanPooling
    assert "_cls: MeanPooling" in str(config)


def test_container_indended_str():
    config = Container(a=3, b=4)
    config_str = config.indented_str(" " * 4)
    assert "    a: 3\n    b: 4\n" in config_str


def test_container_cls_fields():
    from s3prl.nn.pooling import MeanPooling

    config = Container(a=dict(cls=MeanPooling))
    config.cls_fields == [("a.cls", config.a.cls)]


def test_kwds():
    from s3prl.nn.pooling import MeanPooling

    config = Container(a=3, _cls="s3prl.nn.pooling.MeanPooling")
    assert config._cls == MeanPooling
    assert "_cls" not in config.kwds()


def test_container_field():
    eg_field = field(3, "hello", int)
    assert str(eg_field) == "3    [COMMENT] (int) hello"


def test_container_field_with_container():
    config = Container(doc_field=field(3, "the total steps", int))
    assert config.doc_field.value == 3

    new_config = dict(doc_field=4)
    config.override(new_config)
    assert config.doc_field.value == 4

    config.doc_field = 5
    assert config.doc_field.value == 5

    assert "the total steps" in str(config).replace("[SPACE]", " ")

    new_config = dict(doc_field=field(4, "best", int))
    config.override(new_config)
    assert config.doc_field.value == 4
    assert "best" in str(config)

    new_config = Container(
        doc_field=field(7, "coding", int, lambda x: isinstance(x, int))
    )
    config.override(new_config)
    assert config.doc_field.value == 7
    assert "coding" in str(config)

    with pytest.raises(ValueError):
        config.doc_field = 1.3


def test_container_items():
    config = Container(a=3, b=field(0.1, "hello"))
    assert config.b.value == 0.1
    assert config["b"].value == 0.1
    values = list(config.values())
    assert values[-1].value == 0.1
    values = [v[-1] for v in config.items()]
    assert values[-1].value == 0.1


def test_extract_fields():
    config = Container(a=3, b=field(0.1, "hello"))
    assert config.extract_fields().b == 0.1
