import random
from pathlib import Path

import pyarrow as pa
import torch
from tqdm import tqdm

from s3prl.util.benchmark import benchmark

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FEATURE_ROOT = Path("result").resolve()
    FEATURE_NUM = 300
    WAV_NUM = 1000
    WAV_ROOT = FEATURE_ROOT / "wav"
    PARQUET_ROOT = FEATURE_ROOT / "parquet"
    PICKLE_ROOT = FEATURE_ROOT / "pickle"
    MMP_ROOT = FEATURE_ROOT / "mmp"
    FEATHER_ROOT = FEATURE_ROOT / "feather"

    with pa.memory_map(str(FEATURE_ROOT / "feature.arrow")) as sink:
        with pa.ipc.open_file(sink) as reader:
            batches = reader.read_all().to_batches()

    index = list(range(len(batches)))[100:200]
    random.shuffle(index)

    for i in tqdm(index):
        array = batches[i]["feature"].to_numpy(zero_copy_only=False, writable=True)

    index = list(range(len(batches)))[200:300]
    for i in tqdm(index):
        array = batches[i]["feature"].to_numpy(zero_copy_only=False, writable=True)
