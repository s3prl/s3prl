import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.nn as nn
import torch.optim

from s3prl.nn import UtteranceLevel
from s3prl.util.workspace import Checkpoint, Workspace


def test_workspace():
    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        workspace = Workspace(tempdir)

        workspace.set_rank(0)
        stats = dict(a=3, b=4)
        workspace.put(stats, "stats")
        assert workspace["stats"] == stats

        assert "hello" not in workspace
        with pytest.raises(KeyError):
            workspace["hello"]

        model = UtteranceLevel(3, 4, [128])
        workspace["model"] = model
        new_model = workspace["model"]
        assert id(model) != id(new_model)
        assert torch.allclose(
            model.state_dict()["final_proj.weight"].view(-1),
            new_model.state_dict()["final_proj.weight"].view(-1),
        )
        assert "model" in workspace
        assert (workspace / "model.obj").is_file()
        workspace.remove("model")
        assert not (workspace / "model.obj").exists()

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        workspace.put(opt.state_dict(), "optimizer", dtype="pt")
        assert (tempdir / "optimizer.pt").exists()
        new_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        new_opt.load_state_dict(workspace["optimizer"])
        assert opt.state_dict() == new_opt.state_dict()

        assert (workspace / "hello").rank == 0
        workspace.set_rank(1)
        assert (workspace / "hello").rank == 1
        workspace["on_rank_1"] = 3
        assert "on_rank_1" not in workspace


def test_checkpoint():
    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        checkpoint_dir = Checkpoint(tempdir)

        stats = dict(a=3, b=4)
        checkpoint_dir.register_item(stats, "stats", "yaml")
        stats["a"] = 5
        checkpoint_dir.save_to("step-10", "temp")

        loaded_stats = (checkpoint_dir / "step-10" / "temp").get("stats")
        assert loaded_stats == dict(a=5, b=4)


def test_temp_workspace():
    tempdir = Workspace()
    temppath = str(tempdir)
    assert os.path.exists(str(temppath))
    del tempdir
    assert not os.path.exists(str(temppath))


def _get_temp_dirs(true_dir):
    tmp1 = Workspace()
    tmp1["hi"] = 3
    tmp2 = tmp1 / "best"
    tmp2["hi"] = 3
    tmp3 = Workspace(true_dir)
    tmp3["hi"] = 3
    tmp4 = tmp3 / "another"
    tmp4["hi"] = 3
    tmpdirs = [str(tmp1), str(tmp2)]
    truedirs = [str(tmp3), str(tmp4)]
    for folder in [*tmpdirs, *truedirs]:
        assert os.path.exists(folder)
    return tmpdirs, truedirs


def test_workspace_init():
    with TemporaryDirectory() as filepath:
        tmpdirs, truedirs = _get_temp_dirs(filepath)
        for tmpdir in tmpdirs:
            assert not os.path.exists(tmpdir)
        for truedir in truedirs:
            assert os.path.exists(truedir)


def test_workspace_cfg():
    with TemporaryDirectory() as filepath:
        workspace = Workspace(filepath)
        workspace.put_cfg(test_workspace_cfg, dict(a=3, b=4))
        assert (
            workspace / "_cfg" / f"{test_workspace_cfg.__qualname__}.yaml"
        ).is_file()
        assert workspace.get_cfg(test_workspace_cfg) == dict(a=3, b=4)


def test_workspace_key_value_items():
    with TemporaryDirectory() as filepath:
        workspace = Workspace(filepath)
        obj = dict(a=3, b=[1, 2, 3], c=dict(x=1))
        workspace.update(obj)
        for key in obj.keys():
            assert key in workspace
        for value in obj.values():
            assert value in obj.values()
        for key, value in workspace.items():
            assert obj[key] == value


def test_delete():
    with TemporaryDirectory() as filepath:
        workspace = Workspace(filepath)
        workspace["a"] = 3
        del workspace["a"]
        assert not (workspace / "a.txt").exists()


def test_workspace_environ():
    from s3prl.nn import UtteranceLevel

    with TemporaryDirectory() as filepath:
        workspace = Workspace(filepath)
        environ = dict(
            output_size=3,
            categories=[1, 2, 3],
            pred="hello",
            model=UtteranceLevel(3, 4),
            stats=dict(k=4),
        )
        workspace.environ.update(environ)
        assert (workspace / "_environ" / "model.obj").is_file()

        old_model = environ.pop("model")
        new_model = workspace.environ.pop("model")
        assert torch.allclose(
            old_model.state_dict()[list(old_model.state_dict().keys())[0]],
            new_model.state_dict()[list(new_model.state_dict().keys())[0]],
        )
        assert dict(workspace.environ) == environ
        assert (workspace / "_environ" / "output_size.txt").is_file()
        assert (workspace / "_environ" / "categories.pkl").is_file()
        assert (workspace / "_environ" / "pred.txt").is_file()
        assert (workspace / "_environ" / "stats.pkl").is_file()


def test_workspace_link():
    with TemporaryDirectory() as filepath:
        workspace = Workspace(filepath)
        data = dict(a=3, b=4, c=5)
        (workspace / "hello_dir").put(data, "valid_best", "yaml")
        workspace.link_from("temp", (workspace / "hello_dir"), "valid_best")
        assert (workspace / "temp.yaml").is_symlink()
        assert workspace["temp"] == data


def test_workspace_pickle():
    with tempfile.TemporaryDirectory() as tempdir:
        with tempfile.NamedTemporaryFile() as file:
            workspace = Workspace(tempdir)
            torch.save(workspace, file.name)
            workspace = torch.load(file.name)


def test_workspace_rank():
    with tempfile.TemporaryDirectory() as tempdir:
        workspace = Workspace(tempdir)
        feat_dir = workspace / "feat"


def test_workspace_path():
    workspace = Workspace("hello")
    assert str(workspace.resolve()) == str(Path("hello").resolve())
    assert str(workspace) == str(Path("hello"))
