import inspect
import functools
from typing import Any, List, Union
from collections import OrderedDict

import editdistance

from .logdata import Logs
from .container import Container


class GeneralOutput(Container):
    """
    For inherited classes, their __init__ parameters (signature) can help
    autocomplete and serves as the regularization for available key names.
    However, by default the __init__ will lose the information of the
    ordered nature when people instantiation an inherited class with ordered
    dict like:

    child = ChildOutput(input_size=3, output_size=4)

    where input_size should be the first and output_size is the second
    p.s. assume ChildOutput is an inherited class of GeneralOutput

    Hence, the __init_subclass__ of GeneralOutput takes care of this. Any
    child of GeneralOutput enjoy the autocomplete and name regularization
    but in the meanwhile preserve all the functions of GeneralOutput
    especially when accessing values by index.

    See the below Output class for an example.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        cls_init = cls.__init__

        @functools.wraps(cls_init)
        def validated_and_ordered_init(self: __class__, **kwargs):
            cls_init(self, {})

            available_names = list(inspect.signature(cls_init).parameters.keys())
            available_names.remove("self")

            for key in list(kwargs.keys()):
                if key not in available_names:
                    dists = [
                        (name, editdistance.eval(key, name)) for name in available_names
                    ]
                    sorted_names = [
                        name for name, dist in sorted(dists, key=lambda x: x[1])
                    ]
                    raise ValueError(
                        f"'{key}' is not a valid key for Output.\n"
                        f"Consider to use the following existing keys:\n"
                        f"(top-10 closest) {', '.join(sorted_names[:10])}\n"
                        f"Or add a new key to the {cls_init.__module__}.{cls_init.__qualname__}"
                    )
                self.__setitem__(key, kwargs.pop(key))

        cls.__init__ = validated_and_ordered_init

    def cacheable(self):
        output = self.__class__()
        for key, value in self.items():
            if not isinstance(value, Logs):
                output[key] = value
        return output.detach().cpu()


class Output(GeneralOutput):
    def __init__(
        self,
        x=None,
        x_len=None,
        output=None,
        output_len=None,
        hidden_states=None,
        hidden_states_len=None,
        attention_mask=None,
        logs=None,
        loss=None,
        label=None,
        logit=None,
        prediction=None,
        wav=None,
        wav_len=None,
        source=None,
        source_loader=None,
        category=None,
        input_size=None,
        output_size=None,
    ):
        super().__init__()
