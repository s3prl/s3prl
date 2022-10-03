#!/usr/bin/env python3

import argparse
import os
from pathlib import Path


def load_valid_paths():
    with open("./valid_paths.txt", "r") as fp:
        paths = [line.strip() for line in fp if line.strip() != ""]
        return paths


def get_third_party():
    txt_files = list(Path("./requirements").rglob("*.txt"))
    package_list = []
    for file in txt_files:
        with open(file, "r") as fp:
            for line in fp:
                line = line.strip()
                if line == "":
                    continue
                package_list.append(line.split(" ")[0])
    return package_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        type=str,
        nargs="*",
        default=[],
        help="If no file is given, use the files under ./valid_paths.txt",
    )
    parser.add_argument("--check", action="store_true", help="Only checks the files")
    args = parser.parse_args()

    if len(args.files) == 0:
        args.files = load_valid_paths()

    print(f"Formatting files: {args.files}")
    args.files = " ".join(args.files)

    print("Run flake8")
    # stop the build if there are Python syntax errors or undefined names
    os.system(
        f"flake8 {args.files} --count --select=E9,F63,F7,F82 --show-source --statistics"
    )
    # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
    os.system(
        f"flake8 {args.files} --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics"
    )

    print("Run black")
    if args.check:
        os.system(f"black --check {args.files}")
    else:
        os.system(f"black {args.files}")

    print("Run isort")
    third_party = get_third_party()
    third_party = ",".join(third_party)
    if args.check:
        os.system(
            f"isort --profile black --thirdparty {third_party} --check {args.files}"
        )
    else:
        os.system(f"isort --profile black --thirdparty {third_party} {args.files}")

    if args.check:
        print("Successfully passed the format check!")


if __name__ == "__main__":
    main()
