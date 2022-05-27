_key_to_cls = dict()
_cls_to_key = dict()


def put(key: str = "The _cls key name used in cfg"):
    def wrapper(cls_or_func):
        global _key_to_cls, _cls_to_key
        assert key not in _key_to_cls, f"Duplicated key in registry: {key}"
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
