import argparse
import platform

parser = argparse.ArgumentParser()
parser.add_argument("python_version")
args = parser.parse_args()

version = str(platform.python_version())
assert version.startswith(
    args.python_version
), f"expected python version: {args.python_version}, tox executed python version: {version}"
