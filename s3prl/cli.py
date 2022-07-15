import os
import shutil
import logging
import argparse
import importlib

import torch
import torch.distributed as dist

from s3prl.base import registry
from s3prl.util.override import parse_overrides

logger = logging.getLogger(__name__)
LOGGING_FORMAT = "%(levelname)s | %(asctime)s | %(module)s:%(lineno)d | %(message)s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "qualname",
        help="The qualname of the function to use as a binary tool",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help=(
            "The GPU id this process should use while distributed training. "
            "None when not launched by torch.distributed.launch"
        ),
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("-u", "--usage", action="store_true")
    parser.add_argument("-v", "--verbose", default="INFO")
    args, cfg = parser.parse_known_args()

    root_logger = logging.getLogger()
    root_logger.handlers = []
    logging.basicConfig(level=getattr(logging, args.verbose), format=LOGGING_FORMAT)

    func = registry.get(args.qualname)
    cfg = parse_overrides(cfg)

    if args.usage:
        print(f"Documentation of {func.__module__}.{func.__qualname__}\n\n")
        print(func.__doc__)
        return

    if args.refresh and "workspace" in cfg:
        shutil.rmtree(cfg["workspace"], ignore_errors=True)

    local_rank = os.environ.get("LOCAL_RANK") or args.local_rank
    if local_rank is None:
        func(**cfg)
    else:
        # When torch.distributed.launch is used
        torch.distributed.init_process_group("nccl")
        assert dist.is_initialized()
        cfg["device"] = f"cuda:{local_rank}"
        cfg["rank"] = dist.get_rank()
        cfg["world_size"] = dist.get_world_size()
        func(**cfg)


if __name__ == "__main__":
    main()
