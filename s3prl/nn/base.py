from typing import List

import torch.nn as nn
from s3prl import Object, init


class Module(Object, nn.Module):
    @init.method
    def __init__(self):
        super().__init__()
        self._excluded_from_state_dict: List[str] = []

    def exclude_from_state_dict(self, names: List[str]):
        assert isinstance(names, list)
        for name in names:
            module = getattr(self, name)
            assert isinstance(module, nn.Module)
            self._excluded_from_state_dict.append(name)

    def include_to_state_dict(self, names: List[str]):
        assert isinstance(names, list)
        for name in names:
            module = getattr(self, name)
            assert isinstance(module, nn.Module)
            self._excluded_from_state_dict.remove(name)

    def state_dict(self, *args, **kwargs):
        states: dict = super().state_dict(*args, **kwargs)
        for name in self._excluded_from_state_dict:
            module = getattr(self, name)
            excluded_states = module.state_dict(prefix=f"{name}.")
            for key in excluded_states.keys():
                states.pop(key)

        return states

    def load_state_dict(self, state_dict, strict: bool = True):
        temp_holder = {}
        for name in self._excluded_from_state_dict:
            temp_holder[name] = getattr(self, name)
            delattr(self, name)

        super().load_state_dict(state_dict, strict=strict)

        for name, module in temp_holder.items():
            setattr(self, name, module)

    def on_checkpoint(self, checkpoint: dict):
        super().on_checkpoint(checkpoint)
        checkpoint["_excluded_from_state_dict"] = self._excluded_from_state_dict

    def on_from_checkpoint(self, checkpoint: dict):
        super().on_from_checkpoint(checkpoint)
        self._excluded_from_state_dict = checkpoint["_excluded_from_state_dict"]

    def checkpoint(self):
        checkpoint = super().checkpoint()
        checkpoint["state_dict"] = self.state_dict()
        return checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        object = super().from_checkpoint(checkpoint)
        object.load_state_dict(checkpoint["state_dict"])
        return object
