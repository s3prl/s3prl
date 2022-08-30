from s3prl import Container, field
from s3prl.base import fileio

from .configuration import default_cfg


@default_cfg(
    path=field("???", "The item path"),
    interact=field(True, "entering an interactive session after loading the object"),
)
def visualize(**cfg):
    cfg = Container(cfg)
    item = fileio.load(cfg.path)
    print(repr(item))
    if cfg.interact:
        from ipdb import set_trace

        set_trace()
