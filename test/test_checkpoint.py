import tempfile
from pathlib import Path

import pytest

from s3prl.util.workspace import Workspace


@pytest.mark.parametrize(
    "item", [3, 1.333, "hello", dict(a=3, b=4), [3, 4, 0.1], set([5, 7, 7])]
)
@pytest.mark.parametrize("ext", ["txt", "yaml", "pkl", "pt"])
def test_save_load(item, ext):
    with tempfile.TemporaryDirectory() as dirpath:
        workspace = Workspace(dirpath)
        workspace.put(item, "item", ext)
        new_item = workspace["item"]
        assert item == new_item
