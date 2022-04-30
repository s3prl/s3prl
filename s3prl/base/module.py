from collections import namedtuple
from typing import List, OrderedDict, Tuple

import torch.nn as nn

from .object import Object

_EXCLUDED_PREFIX = "_excluded_prefix"


class Module(nn.Module, Object):
    def __init__(self):
        super().__init__()
        self._excluded_from_state_dict: List[str] = []

    def exclude_from_state_dict(self, *names: Tuple[str]):
        for name in names:
            module = getattr(self, name)
            assert isinstance(module, nn.Module)
            self._excluded_from_state_dict.append(name)

    def include_to_state_dict(self, *names: Tuple[str]):
        for name in names:
            module = getattr(self, name)
            assert isinstance(module, nn.Module)
            self._excluded_from_state_dict.remove(name)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        states: dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        if hasattr(self, "_excluded_from_state_dict"):
            if _EXCLUDED_PREFIX not in states:
                states[_EXCLUDED_PREFIX] = []

            for name in self._excluded_from_state_dict:
                module = getattr(self, name)

                substates = OrderedDict()
                substates._metadata = OrderedDict()
                subprefix = prefix + name + "."

                states[_EXCLUDED_PREFIX].append(subprefix)
                module.state_dict(
                    destination=substates,
                    prefix=subprefix,
                    keep_vars=keep_vars,
                )
                for key in [k for k in list(substates.keys()) if k != _EXCLUDED_PREFIX]:
                    states.pop(key)

        return states

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        The public method `load_state_dict` is not recursive. Hence, if a s3prl.Module
        is inside a regular nn.Module, `s3prl.Module.load_state_dict` won't be called.
        However, `s3prl.Module._load_from_state_dict` is recursive hence it is guaranteed
        to be called for all the sub-modules. Hence, to override this private method might
        be the current best solution
        """
        if _EXCLUDED_PREFIX in state_dict and prefix in state_dict[_EXCLUDED_PREFIX]:
            sub_state_dict = self.state_dict(prefix=prefix)
            sub_state_dict.pop(_EXCLUDED_PREFIX)
            state_dict.update(sub_state_dict)

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

        if _EXCLUDED_PREFIX in unexpected_keys:
            unexpected_keys.remove(_EXCLUDED_PREFIX)

    def checkpoint(self):
        checkpoint = super().checkpoint()
        checkpoint["state_dict"] = self.state_dict()
        return checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        object = super().from_checkpoint(checkpoint)
        object.load_state_dict(checkpoint["state_dict"])
        return object
