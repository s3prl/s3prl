import argparse
from s3prl.util.download import _urls_to_filepaths

def print_cache_path(url: str, refresh: bool):
    print(_urls_to_filepaths(url, refresh=refresh))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print_cache_path(args.url, args.refresh)
