import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module_root")
    parser.add_argument("valid_paths")
    args = parser.parse_args()

    with open(args.valid_paths) as file:
        valid_paths = [line.strip() for line in file.readlines()]

    ignored_paths = []
    module_root_name = Path(args.module_root).stem
    for item in os.listdir(args.module_root):
        pattern = f"{module_root_name}/{item}"
        if pattern not in valid_paths:
            ignored_paths.append(pattern)

    print(" ".join(ignored_paths))
