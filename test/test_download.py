import os
import pytest
from pathlib import Path
from multiprocessing import Process

import torch
from s3prl.util.download import _urls_to_filepaths
from subprocess import STDOUT, TimeoutExpired, check_output

TIMEOUT_SECS = 1
URL = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt"


def test_download():
    filepath = Path(_urls_to_filepaths(URL, download=False))
    if filepath.is_file():
        os.remove(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)

    process = Process(target=_urls_to_filepaths, args=(URL,))
    process.start()
    process.join(timeout=TIMEOUT_SECS)

    assert not filepath.is_file()
    filepath = _urls_to_filepaths(URL)
    torch.load(filepath)
