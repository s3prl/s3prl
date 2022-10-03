import logging
import os
from multiprocessing import Process
from pathlib import Path

import torch

from s3prl.util.download import _urls_to_filepaths

logger = logging.getLogger(__name__)
URL = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt"


def _download_with_timeout(timeout: float, num_process: int):
    processes = []
    for _ in range(num_process):
        process = Process(
            target=_urls_to_filepaths, args=(URL,), kwargs=dict(refresh=True)
        )
        process.start()
        processes.append(process)

    exitcodes = []
    for process in processes:
        process.join(timeout=timeout)
        exitcodes.append(process.exitcode)
    assert len(set(exitcodes)) == 1
    exitcode = exitcodes[0]

    if exitcode != 0:
        for process in processes:
            process.terminate()


def test_download():
    filepath = Path(_urls_to_filepaths(URL, download=False))
    if filepath.is_file():
        os.remove(filepath)

    logger.info("This should timeout")
    _download_with_timeout(0.1, 2)
    assert not filepath.is_file(), (
        "The download should failed due to the too short timeout second: 0.1 sec, "
        "and hence there should not be any corrupted (incomplete) file"
    )

    logger.info("This should success")
    _download_with_timeout(None, 2)
    torch.load(filepath, map_location="cpu")
    assert not Path(str(filepath) + ".lock").exists()
