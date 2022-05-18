from time import sleep

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from s3prl import hub
from s3prl.util.benchmark import benchmark


def put_cuda(queue, done):
    model = hub.wav2vec2_large_ll60k().to("cuda:0")
    model.eval()

    for i in tqdm(range(100)):
        with torch.no_grad():
            repre = model([torch.randn(16000 * 13).to("cuda:0") for i in range(16)])[
                "hidden_states"
            ]
            repre = torch.stack(repre, dim=2)

        queue.put(repre)
        del repre
    done.wait()


if __name__ == "__main__":
    ctx = mp.get_context("forkserver")
    queue = ctx.Queue(maxsize=1)
    done = ctx.Event()
    process = ctx.Process(target=put_cuda, args=(queue, done))
    process.start()
    for i in tqdm(range(100)):
        result = queue.get()
        with benchmark("to cuda:1"):
            result_cuda1 = result.to("cuda:1")
        del result
    done.set()
    process.join()
