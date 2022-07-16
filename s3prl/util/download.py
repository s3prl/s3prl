import logging
import shutil
import tempfile
import hashlib
import requests
import subprocess

import torch
from tqdm import tqdm
from pathlib import Path
from filelock import FileLock

logger = logging.getLogger(__name__)


def _requests_get(url: str, filepath: str):
    subprocess.check_call(["wget", f"{url}", "-O", f"{filepath}"])


def _download(filepath: Path, url, refresh):
    if not filepath.is_file() or refresh:
        with tempfile.TemporaryDirectory() as tempdir:
            temppath = Path(tempdir) / filepath.name
            with FileLock(str(temppath) + ".lock"):
                _requests_get(url, temppath)
            filepath.parent.mkdir(exist_ok=True, parents=True)
            shutil.move(temppath, filepath)
    logger.info(f"Using cache found in {filepath}\nfor {url}")


def _urls_to_filepaths(*args, refresh=False, download: bool = True):
    """
    Preprocess the URL specified in *args into local file paths after downloading

    Args:
        Any number of URLs (1 ~ any)

    Return:
        Same number of downloaded file paths
    """

    def _url_to_filepath(url):
        assert isinstance(url, str)
        m = hashlib.sha256()
        m.update(str.encode(url))
        filepath = (
            Path(torch.hub.get_dir())
            / "s3prl"
            / f"{str(m.hexdigest())}.{Path(url).name}"
        )
        if download:
            _download(filepath, url, refresh=refresh)
        return str(filepath.resolve())

    paths = [_url_to_filepath(url) for url in args]
    return paths if len(paths) > 1 else paths[0]
