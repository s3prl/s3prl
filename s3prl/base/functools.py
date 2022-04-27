import logging
import types

logger = logging.getLogger(__name__)


def resolve_qualname(qualname: str, module: types.ModuleType):
    if "<locals>" in qualname:
        logger.error(
            "qualname should not contain '<local>'. "
            "That is, the original function (or class) "
            "should not be defined on-the-fly."
        )
        raise ValueError

    if "." not in qualname:
        return getattr(module, qualname)
    parent, attr = qualname.rsplit(".", maxsplit=1)
    return getattr(resolve_qualname(parent, module), attr)
