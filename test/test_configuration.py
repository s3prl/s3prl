import tempfile

import pytest

from s3prl.util.configuration import default_cfg
from s3prl.util.workspace import Workspace


@default_cfg(
    apple=3,
    b=4,
    c=dict(
        z=9,
        x=10,
    ),
)
def utility(**cfg):
    """utility"""
    return cfg


def test_callable_with_cfg_func():
    assert """utility""" in utility.__doc__
    assert """apple""" in utility.__doc__

    overridden_cfg = utility(
        apple=7,
        c=dict(
            x=20,
        ),
        d=8,
    )
    assert overridden_cfg == dict(
        apple=7,
        b=4,
        c=dict(
            z=9,
            x=20,
        ),
        d=8,
    )

    with pytest.raises(AssertionError):
        utility(cfg=dict(apple=3))


class Utility:
    @default_cfg(
        a=5,
        b=4,
    )
    @classmethod
    def train(cls, **cfg):
        """
        This is Utility.train function
        """
        return cfg


def test_callable_with_cfg_cls():
    overridden_cfg = Utility.train(a=7, z=8)
    assert overridden_cfg == dict(a=7, b=4, z=8)

    with pytest.raises(AssertionError):
        Utility.train(cfg=dict(apple=3))


class ResumableUtility:
    @default_cfg(a=7, c=5, d=4, resume=True, workspace="???")
    @classmethod
    def train(cls, **cfg):
        """
        This is Utility.train function
        """
        return cfg


@pytest.mark.parametrize("cls", [ResumableUtility])
def test_resumable(cls):
    with tempfile.TemporaryDirectory() as tempdir:
        old_cfg = cls.train(workspace=tempdir, z=8, c=1)
        new_cfg = cls.train(
            workspace=tempdir,
        )
        assert old_cfg == new_cfg
        assert Workspace(tempdir).get_cfg(cls.train) == old_cfg
