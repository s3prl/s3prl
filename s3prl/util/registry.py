import logging

_key_to_cls = dict()
_cls_to_key = dict()

logger = logging.getLogger(__name__)


def put(key: str = None):
    """
    key: the _cls field in the config. When key==None, Use the __name__ of the registering object
    """
    _key = key

    def wrapper(cls_or_func):
        key = _key or cls_or_func.__name__
        global _key_to_cls, _cls_to_key
        if key in _key_to_cls:
            logger.warning(
                f"Duplicated registration of the same key name: {key}. "
                f"{_key_to_cls[key]} will be replaced by {cls_or_func}."
            )
        _key_to_cls[key] = cls_or_func
        _cls_to_key[cls_or_func] = key
        return cls_or_func

    return wrapper


def get(key: str = "The _cls name used in cfg"):
    assert key in _key_to_cls, f"{key} not found in the registry"
    return _key_to_cls[key]


def serialize(cls_or_func):
    assert cls_or_func in _cls_to_key, f"{cls_or_func} is not yet registered"
    return _cls_to_key[cls_or_func]


def contains(key_or_cls_or_func):
    return (key_or_cls_or_func in _key_to_cls) or (key_or_cls_or_func in _cls_to_key)
