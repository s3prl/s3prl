import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from s3prl import hub
from s3prl.dataset.base import AugmentedDynamicItemDataset, default_collate_fn
from s3prl.util.benchmark import benchmark

WAV_MIN_SEC = 20
WAV_MAX_SEC = 30
SAMPLE_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_ROOT = Path("/home/leo/features").resolve()
FEATURE_NUM = 300
WAV_NUM = 1000
WAV_ROOT = FEATURE_ROOT / "wav"
PARQUET_ROOT = FEATURE_ROOT / "parquet"
PICKLE_ROOT = FEATURE_ROOT / "pickle"
MMP_ROOT = FEATURE_ROOT / "mmp"
FEATHER_ROOT = FEATURE_ROOT / "feather"


def get_wavs(num: int = 10):
    for i in range(num):
        sec = random.randint(WAV_MIN_SEC, WAV_MIN_SEC)
        samples = sec * SAMPLE_RATE
        yield torch.randn(samples).numpy()


class WavToFeat:
    def __init__(self, upstream: str) -> None:
        self.model = getattr(hub, upstream)().to(device)
        self.model.eval()

    def __call__(self, wav: np.ndarray):
        wav = torch.from_numpy(wav)
        with torch.no_grad():
            wav = wav.to(device)
            repre = self.model([wav])["hidden_states"]
            repre = torch.stack(repre, dim=2).squeeze(0)
        return repre.detach().cpu().numpy()


def get_features(num: int = 10, upstream: str = "wav2vec2_large_ll60k"):
    model = WavToFeat(upstream)
    with torch.no_grad():
        for wav in get_wavs(num):
            repre = model(wav)
            yield repre


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


@pytest.mark.practice
def test_encode_array():
    for feature in get_features(upstream="apc"):
        encoded = encode_array(feature)
        decoded = decode_array(encoded)
        assert np.allclose(feature, decoded)


@pytest.mark.practice
def test_write_wav(num: int = WAV_NUM):
    WAV_ROOT.mkdir(exist_ok=True, parents=True)
    for idx, wav in tqdm(enumerate(get_wavs(num))):
        np.save(WAV_ROOT / f"{idx}.npy", wav)


@pytest.mark.practice
def test_write_parquet():
    PARQUET_ROOT.mkdir(exist_ok=True, parents=True)
    for idx, feature in tqdm(
        enumerate(get_features(FEATURE_NUM, upstream="wav2vec2_large_ll60k"))
    ):
        encoded = encode_array(feature)
        df = pd.DataFrame(data=encoded, columns=["feature"])
        df.to_parquet(PARQUET_ROOT / f"{idx}.parquet")


@pytest.mark.practice
def test_write_feather():
    FEATHER_ROOT.mkdir(exist_ok=True, parents=True)
    for idx, feature in tqdm(
        enumerate(get_features(FEATURE_NUM, upstream="wav2vec2_large_ll60k"))
    ):
        encoded = encode_array(feature)
        df = pd.DataFrame(data=encoded, columns=["feature"])
        df.to_feather(FEATHER_ROOT / f"{idx}.feather")


@pytest.mark.practice
def test_write_pickle():
    PICKLE_ROOT.mkdir(exist_ok=True, parents=True)
    for idx, feature in tqdm(
        enumerate(get_features(FEATURE_NUM, upstream="wav2vec2_large_ll60k"))
    ):
        encoded = encode_array(feature)
        with (PICKLE_ROOT / f"{idx}.pkl").open("wb") as file:
            pickle.dump(encoded, file)


@pytest.mark.practice
def test_write_mmp():
    MMP_ROOT.mkdir(exist_ok=True, parents=True)

    schema = pa.schema([("feature", pa.float32())])
    with pa.OSFile(str(MMP_ROOT / "mmp"), "wb") as source:
        with pa.ipc.new_file(source, schema=schema) as writer:
            for idx, feature in tqdm(
                enumerate(get_features(FEATURE_NUM, upstream="wav2vec2_large_ll60k"))
            ):
                encoded = encode_array(feature)
                batch = pa.record_batch([pa.array(encoded)], schema)
                writer.write(batch)


@pytest.mark.practice
def test_read_mmp():
    with pa.memory_map(str(MMP_ROOT / "mmp"), "rb") as source:
        with pa.ipc.open_file(source) as reader:
            batches = reader.read_all().to_batches()

    big_tensor = torch.zeros(16000 * 5000).cuda()
    indices = list(range(len(batches)))
    random.shuffle(indices)
    for idx in tqdm(indices):
        array = batches[idx]["feature"].to_numpy()
        decoded = decode_array(array)
        tensor = torch.from_numpy(decoded)
        tensor = tensor.clone()
        shape = tensor.shape
        big_tensor[: len(tensor.view(-1))] = tensor.view(-1)
        tensor = big_tensor[: len(tensor.view(-1))].view(shape)


@pytest.mark.practice
def test_read_parquet():
    data = {
        idx: {"path": PARQUET_ROOT / f"{idx % 100}.parquet"}
        for idx in range(FEATURE_NUM * 100)
    }
    dataset = AugmentedDynamicItemDataset(data)
    dataset.add_dynamic_item(
        lambda path: decode_array(pd.read_parquet(path).feature.to_numpy()),
        takes="path",
        provides="array",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=default_collate_fn,
        shuffle=True,
    )
    for batch in tqdm(dataloader):
        batch.to("cuda")


@pytest.mark.practice
def test_read_feather():
    data = {
        idx: {"path": FEATHER_ROOT / f"{idx % 100}.feather"}
        for idx in range(FEATURE_NUM * 100)
    }
    dataset = AugmentedDynamicItemDataset(data)
    dataset.add_dynamic_item(
        lambda path: decode_array(pd.read_feather(path).feature.to_numpy()),
        takes="path",
        provides="array",
    )
    dataloader = DataLoader(
        dataset, batch_size=8, num_workers=8, collate_fn=default_collate_fn
    )
    for batch in tqdm(dataloader):
        batch.to("cuda")


@pytest.mark.practice
def test_read_pickle():
    data = {
        idx: {"path": PICKLE_ROOT / f"{idx % 100}.pkl"}
        for idx in range(FEATURE_NUM * 100)
    }
    dataset = AugmentedDynamicItemDataset(data)

    def read_pickle(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    dataset.add_dynamic_item(
        lambda path: decode_array(read_pickle(path)),
        takes="path",
        provides="array",
    )
    dataloader = DataLoader(
        dataset, batch_size=8, num_workers=4, collate_fn=default_collate_fn
    )
    for batch in tqdm(dataloader):
        batch.to("cuda")


@pytest.mark.practice
def test_hf_dataset():
    from datasets.arrow_dataset import Dataset

    dataset = Dataset.from_dict({"id": list(range(5))})

    def add_wav(x):
        x["wav"] = list(get_wavs(num=1))[0]
        return x

    def add_feat(x):
        x["feat"] = list(get_features(num=1))[0]
        return x

    dataset = dataset.map(add_feat)

    with benchmark("load_mem_mapped"):
        dataset[0]


@pytest.mark.practice
def test_load_parquet():
    random.seed(1)
    torch.manual_seed(1)

    test_write_parquet()

    data = {
        idx: {"path": PARQUET_ROOT / f"{idx % 100}.parquet"}
        for idx in range(FEATURE_NUM * 100)
    }
    dataset = AugmentedDynamicItemDataset(data)
    dataset.add_dynamic_item(
        lambda path: decode_array(pd.read_parquet(path).feature.to_numpy()),
        takes="path",
        provides="array",
    )

    with benchmark("parquet load"):
        array = dataset[0]["array"]
    with benchmark("to cuda"):
        array = torch.from_numpy(array).cuda()
    array.sum(axis=-1)


@pytest.mark.practice
def test_on_the_fly():
    random.seed(1)
    torch.manual_seed(1)

    test_write_wav(num=FEATURE_NUM)
    data = {
        idx: {"path": WAV_ROOT / f"{idx % 100}.npy"} for idx in range(FEATURE_NUM * 100)
    }
    dataset = AugmentedDynamicItemDataset(data)
    dataset.add_dynamic_item(
        lambda path: np.load(path),
        takes="path",
        provides="wav",
    )

    model = WavToFeat("wav2vec2_large_ll60k")

    for item in tqdm(dataset):
        wav = item["wav"]
        array = model(wav)
        tensor = torch.from_numpy(array).cuda()
