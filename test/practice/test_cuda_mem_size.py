import logging
import random
from pathlib import Path
from time import time

import pyarrow as pa
import torch
from tqdm import tqdm

from s3prl import hub
from s3prl.util.benchmark import benchmark

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    logging.basicConfig(level=logging.INFO)
    logger.info("start")

    root = Path("sdb/")
    total_size = 100
    bucket_size = 8

    model = hub.wav2vec2_large_ll60k().to("cuda:0")
    model.eval()

    bucket_repre = []
    for i in tqdm(range(total_size)):
        with torch.no_grad():
            repre = model([torch.randn(16000 * 10).to("cuda:0") for i in range(16)])[
                "hidden_states"
            ]
            repre = torch.stack(repre, dim=2).detach().cpu()
            torch.save(repre, root / f"{i}.pth")

            bucket_repre.append(repre)
            if len(bucket_repre) == bucket_size:
                torch.save(torch.stack(bucket_repre, dim=0), root / f"{i}.bucket.pth")
                bucket_repre = []

    start = time()
    indices = list(range(total_size))
    random.shuffle(indices)
    for i in tqdm(indices):
        torch.load(root / f"{i}.pth")
    print("small chunks", time() - start)

    start = time()
    indices = [i + bucket_size - 1 for i in list(range(0, total_size, bucket_size))]
    random.shuffle(indices)
    for i in tqdm(indices):
        path = root / f"{i}.bucket.pth"
        if path.is_file():
            torch.load(path)
    print("large chunk", time() - start)
