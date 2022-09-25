"""
Thread-safe file downloading and cacheing

Authors
  * Leo 2022
"""

import hashlib
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

from filelock import FileLock
from tqdm import tqdm

logger = logging.getLogger(__name__)


_download_dir = Path.home() / ".cache" / "s3prl" / "download"

__all__ = [
    "get_dir",
    "set_dir",
    "download",
    "urls_to_filepaths",
]


def get_dir():
    _download_dir.mkdir(exist_ok=True, parents=True)
    return _download_dir


def set_dir(d):
    global _download_dir
    _download_dir = Path(d)


def _download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """
    This function is not thread-safe. Please ensure only a single
    thread or process can enter this block at the same time
    """

    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()

        tqdm.write(f"Downloading: {url}", file=sys.stderr)
        tqdm.write(f"Destination: {dst}", file=sys.stderr)
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest
                    )
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def _download(filepath: Path, url, refresh: bool, new_enough_secs: float = 2.0):
    """
    If refresh is True, check the latest modfieid time of the filepath.
    If the file is new enough (no older than `new_enough_secs`), than directly use it.
    If the file is older than `new_enough_secs`, than re-download the file.
    This function is useful when multi-processes are all downloading the same large file
    """

    Path(filepath).parent.mkdir(exist_ok=True, parents=True)

    lock_file = Path(str(filepath) + ".lock")
    logger.info(f"Requesting URL: {url}")

    with FileLock(str(lock_file)):
        if not filepath.is_file() or (
            refresh and (time.time() - os.path.getmtime(filepath)) > new_enough_secs
        ):
            _download_url_to_file(url, filepath)

    logger.info(f"Using URL's local file: {filepath}")
    try:
        lock_file.unlink()
    except FileNotFoundError:
        pass


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
        filepath = get_dir() / f"{str(m.hexdigest())}.{Path(url).name}"
        if download:
            _download(filepath, url, refresh=refresh)
        return str(filepath.resolve())

    paths = [_url_to_filepath(url) for url in args]
    return paths if len(paths) > 1 else paths[0]


download = _download
urls_to_filepaths = _urls_to_filepaths
