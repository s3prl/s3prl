from pathlib import Path

from . import hub

with (Path(__file__).parent / "version.txt").open() as file:
    __version__ = file.read()
