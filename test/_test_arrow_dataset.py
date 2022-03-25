import os
import pyarrow as pa
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
from time import time
from tqdm import tqdm
import numpy as np

ROOT = Path("result/")
NUM_BATCH = 500
PT = ROOT / "pt"
PKL = ROOT / "pkl"
NP = ROOT / "np"
H5 = ROOT / "h5"
TIMES = 5000


class ArrowDataset(Dataset):
    def __init__(self, arrow_path: str) -> None:
        super().__init__()
        self.arrow_path = arrow_path
        self.source = pa.memory_map(str(arrow_path), "rb")
        self.reader = pa.ipc.open_file(self.source)
        self.num_batches = len(self.reader.read_all().to_batches())

    def __getitem__(self, index):
        index = index % self.num_batches
        return self._read_batch(index)

    def _read_batch(self, index):
        tensor_np = self._get_batch(index)[0].to_numpy()
        tensor = torch.from_numpy(tensor_np)
        return tensor

    def _get_batch(self, index):
        return self.reader.get_batch(index)

    def __len__(self):
        return self.num_batches * TIMES

    def __del__(self):
        self.source.close()


class NPDataset(Dataset):
    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = Path(root)
        self.num = len(os.listdir(self.root))

    def __getitem__(self, index):
        index = index % self.num
        return np.load(self.root / f"{index}.np.npy")

    def __len__(self):
        return self.num * TIMES


class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = torch.randn(100000, 100)

    def __getitem__(self, index):
        pid = os.getpid()
        self.data[index] = torch.randn(100)
        print(pid, self.data[0].numpy()[:5])
        return self.data[index]

    def __len__(self):
        return 100000


def test_testdataset():
    dataset = TestDataset()
    dataloader = DataLoader(
        dataset, num_workers=2, batch_size=20, collate_fn=lambda x: x
    )
    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True):
        pass


class FastArrowDataset(Dataset):
    def __init__(self, arrow_path: str) -> None:
        super().__init__()
        self.arrow_path = arrow_path
        with pa.memory_map(str(arrow_path), "rb") as source:
            with pa.ipc.open_file(source) as reader:
                self.num_batches = len(reader.read_all().to_batches())

    def __getitem__(self, index):
        index = index % self.num_batches
        tensor = self._read_batch(index)
        return tensor

    def _read_batch(self, index):
        tensor_np = self._get_batch(index)[0].to_numpy()
        tensor = torch.from_numpy(tensor_np)
        return tensor

    def _get_batch(self, index):
        if not hasattr(self, "batches"):
            with pa.memory_map(str(self.arrow_path), "rb") as source:
                with pa.ipc.open_file(source) as reader:
                    self.batches = reader.read_all().to_batches()
        return self.batches[index]

    def __len__(self):
        return self.num_batches * TIMES

    def __del__(self):
        self.source.close()


def test_arrow_dataset():
    start = time()

    dataset = ArrowDataset(ROOT / "feature.arrow")
    dataloader = DataLoader(
        dataset, num_workers=0, batch_size=20, collate_fn=lambda x: x
    )
    for i in range(len(dataset)):
        start = time()
        batch = dataset[i]
        from ipdb import set_trace
        set_trace()
        print(time() - start)

    print("all", time() - start)


def test_fast_arrow_dataset():
    start = time()

    dataset = FastArrowDataset(ROOT / "feature.arrow")
    dataloader = DataLoader(
        dataset, num_workers=0, batch_size=20, collate_fn=lambda x: x, shuffle=True
    )
    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True):
        pass

    print("all", time() - start)


def test_np_dataset():
    start = time()

    dataset = NPDataset(NP)
    dataloader = DataLoader(
        dataset, num_workers=12, batch_size=20, collate_fn=lambda x: x
    )
    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True):
        pass

    print("all", time() - start)


if __name__ == "__main__":
    dataset = ArrowDataset(ROOT / "feature.arrow")
    dataloader = DataLoader(
        dataset, num_workers=0, batch_size=20, collate_fn=lambda x: x
    )
    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True):
        pass
