import os
import shutil
import logging
import argparse
import importlib

import torch
import torch.distributed as dist
from s3prl.util.override import parse_overrides
from s3prl.problem.base import all_problems

logger = logging.getLogger(__name__)
LOGGING_FORMAT = "%(levelname)s | %(asctime)s | %(module)s:%(lineno)d | %(message)s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "qualname",
        help="The qualname of the function to use as a binary tool",
    )
    parser.add_argument(
        "-m",
        "--module",
        help="The module of the qualname. "
        "If not given, find the qualname object from the problem package.",
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

    if args.module is not None:
        module = importlib.import_module(args.module)
        target = module
        for name in args.qualname.split("."):
            target = getattr(target, name)
    else:
        classname, stepname = args.qualname.split(".")
        cls = all_problems[classname]
        target = getattr(cls, stepname)

    cfg = parse_overrides(cfg)

    if args.usage:
        print(f"Documentation of {target.__module__}.{target.__qualname__}\n\n")
        print(target.__doc__)
        return

    if args.refresh and "workspace" in cfg:
        shutil.rmtree(cfg["workspace"], ignore_errors=True)

    local_rank = os.environ.get("LOCAL_RANK") or args.local_rank
    if local_rank is None:
        target(**cfg)
    else:
        # When torch.distributed.launch is used
        torch.distributed.init_process_group("nccl")
        assert dist.is_initialized()
        cfg["device"] = f"cuda:{local_rank}"
        cfg["rank"] = dist.get_rank()
        cfg["world_size"] = dist.get_world_size()
        target(**cfg)


if __name__ == "__main__":
    main()
