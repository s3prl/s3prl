import multiprocessing as mp
from time import sleep

import torch
from tqdm import tqdm

from s3prl import hub


def put_cuda(conn, done):
    model = hub.wav2vec2_large_ll60k().to("cuda:0")
    model.eval()

    for i in tqdm(range(100)):
        conn.recv()
        with torch.no_grad():
            repre = model([torch.randn(16000 * 14).to("cuda:0") for i in range(16)])[
                "hidden_states"
            ]
        conn.send(repre)
        del repre
    done.wait()


if __name__ == "__main__":
    ctx = mp.get_context("forkserver")
    main, child = ctx.Pipe()
    done = ctx.Event()
    process = ctx.Process(target=put_cuda, args=(child, done))
    process.start()
    main.send(None)
    for i in tqdm(range(100)):
        result = main.recv()
        main.send(None)
        del result
    done.set()
    process.join()
