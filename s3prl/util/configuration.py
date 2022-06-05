"""

**Magic keys**

.. note::

    Magic key is an advanced usage. Please first see the Usage of `default_cfg` or `override_parent_cfg` for the basic context

Magic keys given to the `default_cfg` or `override_parent_cfg` will cause some automatic behavior. \
    These behavior can be handy for many machine learning use cases. Current the supported magic keys \
    contain:

- **workspace (path):** Will save the input config of the `function` to `path`/_cfg/`function.__qualname__`.yaml

- **resume (bool):** `workspace` must be presented in the cfg. When True, will load the config pre-saved as the default config

- **stage_{int} (any dict):** The dict must contain a `_method` key, indicating the classmethod to be ran. This magic key helps copying the default cfg of each stage classmethod into the current dict.

"""

import abc
import functools
import logging
import re
from typing import Any, Callable

from s3prl import Container, field
from s3prl.util.doc import _longestCommonPrefix
from s3prl.base.container import _qualname_to_cls
from s3prl.util import registry

from .workspace import Workspace

logger = logging.getLogger(__name__)


def _preprocess_doc(obj):
    obj.__doc__ = obj.__doc__ or ""
    doc = obj.__doc__
    lines = [line for line in doc.split("\n") if len(line) > 0]
    indent = _longestCommonPrefix(lines)
    if len(indent) == 0:
        indent = " " * 4
    return indent


def _add_doc(obj, doc, last=True):
    indent = _preprocess_doc(obj)
    doc = "\n".join([f"{indent}{line}" for line in doc.split("\n")])

    if last:
        obj.__doc__ = f"{obj.__doc__}\n{doc}"
    else:
        obj.__doc__ = f"{doc}\n{obj.__doc__}"


class _CallableWithConfig:
    @property
    @abc.abstractmethod
    def default_cfg(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_classmethod(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def add_default_cfg_doc(self):
        indent = _preprocess_doc(self)

        start = f"\n\n{indent}- *Necessary Config:*\n\n"
        end = f"\n{indent}========================"
        default_cfg: Container = self.default_cfg
        unfilled_fields = default_cfg.unfilled_fields()
        if len(unfilled_fields) > 0:
            content = f"{indent}.. code-block:: yaml\n\n{default_cfg.unfilled_fields().indented_str(indent * 2)}\n"
            content = re.sub("\[COMMENT\]", rf"#", content)
            content = re.sub("\[DOT\]", ":", content)
            content = re.sub("\[SPACE\]", " ", content)
            content = re.sub("\[NEWLINE\]", rf"\n{indent * 2}  # ", content)
        else:
            content = f"{indent}**None**\n\n"

        clses = []
        for name, value in default_cfg.cls_fields():
            if isinstance(value, field):
                value = value.value
            if value == Container.UNFILLED_PATTERN:
                continue
            if isinstance(value, str):
                try:
                    value = _qualname_to_cls(value)
                except:
                    try:
                        value = registry.get(value)
                    except:
                        pass
            if isinstance(value, type) and value not in clses:
                clses.append(value)

        content += f"{indent}- *All Config (default):*\n\n{indent}{', '.join([f':py:obj:`~{value.__module__}.{value.__qualname__}`' for value in clses])}\n\n{indent}.. code-block:: yaml\n\n{default_cfg.indented_str(indent * 2)}\n"
        content = re.sub("\[COMMENT\]", rf"#", content)
        content = re.sub("\[DOT\]", ":", content)
        content = re.sub("\[SPACE\]", " ", content)
        content = re.sub("\[NEWLINE\]", rf"\n{indent * 2}  # ", content)
        if re.search(rf"{start}.+{end}", self.__doc__) is not None:
            self.__doc__ = re.sub(
                rf"{start}.+{end}", f"{start}{content}{end}", self.__doc__
            )
        else:
            self.__doc__ += f"{start}{content}{end}"

    def __call__(self, *args, **cfg) -> Any:
        assert "cfg" not in cfg, "Please decompose your input dict by **cfg"
        cfg = Container(cfg).extract_fields()
        all_cfg = self.default_cfg.clone().extract_fields()
        all_cfg.override(cfg)

        if "workspace" in all_cfg:
            assert (
                cfg.workspace != cfg.UNFILLED_PATTERN
            ), "You should fill the 'workspace' field"
            workspace = Workspace(cfg.workspace)

            saved_cfg = workspace.get_cfg(self)
            if all_cfg.get("resume", False) and saved_cfg is not None:
                logger.info(f"Load config from {workspace}")
                all_cfg.override(saved_cfg)
                all_cfg.override(cfg)

        non_stage_cfgs = Container()
        for key in all_cfg.keys():
            match = re.search("stage", key)  # include start_stage and final_stage
            if match is None:
                non_stage_cfgs[key] = all_cfg[key]
        for key in all_cfg.keys():
            match = re.search("stage_(\d)+", key)
            if match is not None:
                all_cfg[key].override(non_stage_cfgs)

        all_cfg.check_no_unfilled_field()

        if "workspace" in all_cfg:
            workspace.set_rank(cfg.get("rank", 0))
            workspace.put_cfg(self, all_cfg)
            logger.info(f"Save config to {workspace}")

        if isinstance(self, _CallableWithDefaultConfig):
            logger.info(f"\n\nFINAL CONFIG:\n\n{str(all_cfg)}")

            if "workspace" in all_cfg:
                log_file = str(
                    Workspace(all_cfg.workspace).get_log_file(self).resolve()
                )
                root_log = logging.getLogger()
                formatter = logging.Formatter(
                    f"[%(levelname)s] RANK {all_cfg.get('rank', 0)} %(asctime)s (%(module)s.%(funcName)s:%(lineno)d): %(message)s"
                )
                fileHandler = logging.FileHandler(log_file)
                fileHandler.setFormatter(formatter)
                root_log.addHandler(fileHandler)

        return self._func(*args, **all_cfg)


class _CallableWithDefaultConfig(_CallableWithConfig):
    def __init__(self, caller: Callable, default_cfg: dict) -> None:
        self._caller = caller
        if self.is_classmethod:
            self._func = caller.__func__
        else:
            self._func = self._caller
        self._default_cfg = Container(default_cfg)
        functools.update_wrapper(self, self._func)
        self.add_default_cfg_doc()

    @property
    def is_classmethod(self):
        return isinstance(self._caller, classmethod)

    @property
    def default_cfg(self) -> Container:
        return self._default_cfg.readonly()

    def __set_name__(self, owner, name):
        if hasattr(self._func, "__set_name__"):
            self._func.__set_name__(owner, name)

        if isinstance(self._caller, classmethod):
            setattr(owner, name, classmethod(self))

        default_cfg = self.default_cfg.clone()
        stage_cfgs = []
        for key in default_cfg.keys():
            match = re.search("stage_(\d)+", key)
            if match is not None:
                step_id = int(match.groups()[0])
                if field.sanitize(default_cfg[key]["_method"]) != "???":
                    stage_cfgs.append((step_id, key, default_cfg[key]))
        stage_cfgs.sort(key=lambda x: x[0])
        if len(stage_cfgs) > 0:
            for step_id, key, stage_cfg in stage_cfgs:
                stage_method = getattr(owner, field.sanitize(stage_cfg["_method"]))
                stage_default_cfg = stage_method.default_cfg.clone()
                stage_default_cfg.override(stage_cfg)
                default_cfg[key] = stage_default_cfg

        self._default_cfg = default_cfg
        functools.update_wrapper(self, self._func)
        self.add_default_cfg_doc()


def default_cfg(**cfg):
    """
    **Usage**

    Wrap a classmethod to define the default values of **cfg, and support dictionary overridding, \
        which is the common case for machine learning hyper-parameter tuning

    .. code-block:: python

        @default_cfg(
            a=3,
            b=dict(
                x=7,
                y=8,
            )
        )
        def train(cls, **cfg):
            assert "a" in cfg
            assert "x" in cfg["b"]
            assert "y" in cfg["b"]

    You can then call `cls.train`:

    .. code-block:: python

        cls.train(a="hello", b=dict(y=9, z=10))

    The `cls.train` method will then get the follow final cfg:

    .. code-block:: python

        cfg = {
            "a": "hello",
            "b": {
                "x": 7,
                "y": 9,
                "z": 10,
            }
        }

    .. rubric:: Auto-documenting

    The default config (dict) will be automatically documented in the method's __doc__
    in RST syntax as a yaml code-block. To add the description for each value, you can
    use the `field` to wrap the value:

    .. code-block:: python

        from s3prl import field

        @default_cfg(
            a=field(3, "This is a doc message", int),
        )
        def train(cls, **cfg):
            assert "a" in cfg

    Then, the __doc__ will be added with

    .. code-block:: yaml

        a: 3    # (int), This is a doc message

    """

    def wrapper(caller):
        wrapped_caller = _CallableWithDefaultConfig(caller, cfg)
        return wrapped_caller

    return wrapper


class _CallableWithOverrideParentConfig(_CallableWithConfig):
    def __init__(self, caller: Callable, override_cfg: dict) -> None:
        self._caller = caller
        assert isinstance(caller, classmethod)
        self._func = caller.__func__
        self._override_cfg = Container(override_cfg)

    @property
    def is_classmethod(self):
        return True

    def __set_name__(self, owner, name):
        if hasattr(self._func, "__set_name__"):
            self._func.__set_name__(owner, name)

        if isinstance(self._caller, classmethod):
            setattr(owner, name, classmethod(self))

        self._parent_method = getattr(super(owner, owner), name)
        overridden_default_cfg: Container = self._parent_method.default_cfg.clone()
        overridden_default_cfg.override(self._override_cfg)

        stage_cfgs = []
        for key in overridden_default_cfg.keys():
            match = re.search("stage_(\d)+", key)
            if match is not None:
                step_id = int(match.groups()[0])
                if field.sanitize(overridden_default_cfg[key]["_method"]) != "???":
                    stage_cfgs.append((step_id, key, overridden_default_cfg[key]))
        stage_cfgs.sort(key=lambda x: x[0])
        if len(stage_cfgs) > 0:
            for step_id, key, stage_cfg in stage_cfgs:
                stage_method = getattr(owner, field.sanitize(stage_cfg["_method"]))
                stage_default_cfg = stage_method.default_cfg.clone()
                stage_default_cfg.override(stage_cfg)
                overridden_default_cfg[key] = stage_default_cfg

        self._default_cfg = overridden_default_cfg
        functools.update_wrapper(self, self._func)
        self.add_default_cfg_doc()

        doc = f"\n\n.. hint::\n\n    Run the following stages sequentially. Please refer to the documentation of each stage.\n"
        doc += "    Each stage has its config, however, the top-level options (like to most outer 'workspace' option) will be automatically "
        doc += "    populated to each stage config. Hence, by simply assigin the most outer workspace, each stage's workspace is assigned "
        doc += "    with the same value. You can still set each stage's option by `stage_0=dict(workspace='specific_path')`, the stage-specific "
        doc += "    value will override the auto-assigned top-level value.\n\n"

        if len(stage_cfgs) > 0:
            for step_id, key, stage_cfg in stage_cfgs:
                stage_method = getattr(owner, field.sanitize(stage_cfg["_method"]))
                doc += f"    |  **stage {step_id}:** :py:obj:`~{stage_method.__module__}.{stage_method.__qualname__}`\n"

            _add_doc(self, doc, last=False)

        doc = f"\n\n.. note::\n\n    Parent method: :py:obj:`~{self._parent_method.__module__}.{self._parent_method.__qualname__}`"
        _add_doc(self, doc, last=False)

    @property
    def default_cfg(self) -> Container:
        return self._default_cfg.readonly()


def override_parent_cfg(**cfg):
    """
    **Usage**

    This has the simliar usage as the `default_cfg`, while it first takes the same-name parent method's \
        default config as its default config, and then override it with the overridding cfg.

    .. code-block:: python

        @override_parent_cfg(
            a=100,
        )
        def train(cls, **cfg):
            assert "a" in cfg
            assert "x" in cfg["b"]
            assert "y" in cfg["b"]

    You can then call `cls.train`:

    .. code-block:: python

        cls.train()

    The `cls.train` method will then get the follow final cfg, taking the default config \
        from either the `override_parent_cfg` or `default_cfg` of its `parent.train` method

    .. code-block:: python

        cfg = {
            "a": 100,
            "b": {
                "x": 7,
                "y": 8,
            }
        }

    """

    def wrapper(caller):
        wrapped_caller = _CallableWithOverrideParentConfig(caller, cfg)
        return wrapped_caller

    return wrapper
