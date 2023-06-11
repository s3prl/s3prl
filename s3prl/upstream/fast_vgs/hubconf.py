import logging
import os
import shutil
from pathlib import Path

from s3prl.util.download import urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)


def fast_vgs_plus(**kwargs):
    tar_file: Path = Path(urls_to_filepaths("https://huggingface.co/s3prl/fast_vgs_plus/resolve/main/fast-vgs-plus-coco.zip"))
    target_dir = Path(tar_file.parent) / tar_file.stem
    if not target_dir.is_dir():

        os.system(f"unzip {tar_file} -d {target_dir}")
    kwargs["ckpt"] = target_dir / "fast-vgs-plus-coco"
    return _UpstreamExpert(**kwargs)

