import logging

logger = logging.getLogger(__name__)

def print_dict(obj, indent=0):
    if isinstance(obj, dict):
        print_dict("dict", indent=indent)
        for idx, (key, value) in enumerate(obj.items()):
            print_dict(idx, indent=indent)
            print_dict(key, indent=indent)
            print_dict(value, indent=indent+1)

    elif isinstance(obj, (list, tuple)):
        print_dict("array", indent=indent)
        for idx, value in enumerate(obj):
            print_dict(idx, indent=indent)
            print_dict(value, indent=indent+1)

    else:
        logger.debug(f"{' ' * indent * 4 + str(obj)}")
