import inspect
import logging
import re
import types

import torch

import s3prl

from . import init

logger = logging.getLogger(__name__)


class Checkpoint:
    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.__init__ = init.method(cls.__init__)

        for funcname in ["on_checkpoint", "on_from_checkpoint"]:
            func: types.FunctionType = getattr(cls, funcname)
            if func != getattr(__class__, funcname):
                assert (
                    re.search(f"super.+{func.__name__}", inspect.getsource(func))
                    is not None
                ), f"You should call parent's {func.__name__} in {func.__module__}.{func.__qualname__}"

    def on_checkpoint(self, checkpoint: dict):
        assert isinstance(checkpoint, dict)
        checkpoint.update(
            dict(
                pytorch_version=torch.__version__,
                s3prl_version=s3prl.__version__,
                init_configs=s3prl.init.serialize(self),
            )
        )

    def on_from_checkpoint(self, checkpoint: dict):
        assert isinstance(checkpoint, dict)
        for package in ["pytorch_version", "s3prl_version"]:
            logger.info(f"{package}: {checkpoint[package]}")

    def checkpoint(self):
        checkpoint = dict()
        self.on_checkpoint(checkpoint)
        return checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        if "state_dict" in checkpoint and not issubclass(cls, s3prl.Module):
            return s3prl.Module.from_checkpoint(checkpoint)

        init_configs = checkpoint["init_configs"]
        object = init.deserialize(init_configs)
        object.on_from_checkpoint(checkpoint)
        return object

    def save_checkpoint(self, path: str):
        # TODO: More sophisticated saving. E.g. different objects use different saver
        # Might be saving to a directory instead of just a file
        torch.save(self.checkpoint(), path)

    @classmethod
    def load_checkpoint(cls, path: str, override: dict = None):
        # TODO: More sophisticated loading. E.g. different objects use different loader
        # Might be saving to a directory instead of just a file
        checkpoint: dict = torch.load(str(path))
        return cls.from_checkpoint(checkpoint)
