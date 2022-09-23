from pathlib import Path

with (Path(__file__).parent.resolve() / "version.txt").open() as file:
    __version__ = file.read().strip()
