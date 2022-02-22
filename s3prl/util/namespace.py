from s3prl import Output

import torch


def detach_namespace(namespace):
    detached = Output()
    for entry in dir(namespace):
        value = getattr(namespace, entry)
        if isinstance(entry, torch.Tensor):
            setattr(entry, value.detach())
    return detached
