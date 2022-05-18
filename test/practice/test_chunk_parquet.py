import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from tqdm import tqdm

from s3prl import hub
from s3prl.util.benchmark import benchmark

logger = logging.getLogger(__name__)


def encode_array(array: np.ndarray):
    shape = array.shape
    metadata = np.array([len(shape), *shape]).astype(array.dtype)
    metadata_with_array = np.concatenate([metadata, array.reshape(-1)])
    return metadata_with_array


def decode_array(metadata_with_array: np.ndarray):
    dim = round(metadata_with_array[0])
    shape = [round(size) for size in metadata_with_array[1 : 1 + dim]]
    array = metadata_with_array[1 + dim :].reshape(shape)
    return array


def write_parquet(array, filepath: str, compression: str = "gzip"):
    if isinstance(array, torch.Tensor):
        array = array.numpy()

    encoded = encode_array(array)
    df = pd.DataFrame(data=encoded, columns=["feature"])
    if ".parquet" not in str(filepath):
        filepath = f"{filepath}.parquet"
    df.to_parquet(filepath, compression=compression)


def read_parquet(filepath: str):
    if ".parquet" not in str(filepath):
        filepath = f"{filepath}.parquet"
    array = decode_array(pd.read_parquet(filepath).feature.to_numpy())
    return array


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/leo/sdb/features")
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--skip_write", action="store_true")
    parser.add_argument("--total_size", type=int, default=10)
    parser.add_argument("--bucket_size", type=int, default=5)
    parser.add_argument("--samples", type=int, default=16000 * 10)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_write:
        model = hub.wav2vec2_large_ll60k().to("cuda:0")
        model.eval()

        bucket_repre = []
        for i in tqdm(range(args.total_size)):
            with torch.no_grad():
                repre = model(
                    [
                        torch.randn(args.samples).to("cuda:0")
                        for i in range(args.batch_size)
                    ]
                )["hidden_states"]
                repre = torch.stack(repre, dim=2).detach().cpu()
                write_parquet(repre, root / f"{i}")

                bucket_repre.append(repre)
                if len(bucket_repre) % args.bucket_size == 0:
                    repre = torch.stack(bucket_repre, dim=0)
                    write_parquet(
                        repre, root / f"{i}.bucket", compression=args.compression
                    )
                    bucket_repre = []

    indices = list(range(0, args.total_size, args.bucket_size))
    indices = [i + args.bucket_size - 1 for i in indices]
    random.shuffle(indices)
    for i in tqdm(indices, desc="load large parquet files"):
        array = read_parquet(root / f"{i}.bucket")
        print(f"array size: {pa.serialize(array).to_buffer().size / (1024 ** 3)}")

    # indices = list(range(args.total_size))
    # random.shuffle(indices)
    # for i in tqdm(indices, desc="load small parquet files"):
    #     read_parquet(root / f"{i}")
