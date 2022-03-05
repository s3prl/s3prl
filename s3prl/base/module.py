from collections import namedtuple
from typing import List, OrderedDict, Tuple

import torch.nn as nn
from . import Object

_EXCLUDED_KEY_SUFFIX = "_excluded_key"


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


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

    def get_extra_state(self):
        return dict(
            _excluded_from_state_dict=self._excluded_from_state_dict,
        )

    def set_extra_state(self, state):
        self._excluded_from_state_dict = state["_excluded_from_state_dict"]

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        states: dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        if hasattr(self, "_excluded_from_state_dict"):
            if _EXCLUDED_KEY_SUFFIX not in states:
                states[_EXCLUDED_KEY_SUFFIX] = []

            for name in self._excluded_from_state_dict:
                module = getattr(self, name)

                substates = OrderedDict()
                substates._metadata = OrderedDict()
                module.state_dict(
                    destination=substates,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )
                for key in [
                    k for k in list(substates.keys()) if _EXCLUDED_KEY_SUFFIX not in k
                ]:
                    states[_EXCLUDED_KEY_SUFFIX].append(key)
                    states.pop(key)

        return states

    def load_state_dict(self, state_dict, strict: bool = True):
        missing_keys, unexpected_keys = super().load_state_dict(
            state_dict, strict=False
        )
        unexpected_keys.remove(_EXCLUDED_KEY_SUFFIX)
        excluded_keys = state_dict.pop(_EXCLUDED_KEY_SUFFIX)
        missing_keys = [m for m in missing_keys if m not in excluded_keys]

        error_msgs: List[str] = []
        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def checkpoint(self):
        checkpoint = super().checkpoint()
        checkpoint["state_dict"] = self.state_dict()
        return checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        object = super().from_checkpoint(checkpoint)
        object.load_state_dict(checkpoint["state_dict"])
        return object
