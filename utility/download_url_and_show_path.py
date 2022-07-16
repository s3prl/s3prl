import argparse

from s3prl.util.download import _urls_to_filepaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    args = parser.parse_args()

    print(_urls_to_filepaths(args.url))
