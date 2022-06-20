from genericpath import exists
import os
from collections import defaultdict
from pathlib import Path

from filelock import FileLock
from librosa.util import find_files
from joblib import Parallel, delayed
from tqdm import tqdm

from s3prl import Output, cache, Container
from s3prl.base.cache import _cache_root
from s3prl.util import registry

from .base import Corpus

SPLIT_FILE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"


class VoxCeleb1SID(Corpus):
    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        self.dataset_root = Path(dataset_root).resolve()

        uid2split = self._get_standard_usage(self.dataset_root)
        self._split2uids = defaultdict(list)
        for uid, split in uid2split.items():
            self._split2uids[split].append(uid)

        uid2wavpath = self._find_wavs_with_uids(
            self.dataset_root, sorted(uid2split.keys()), n_jobs=n_jobs
        )
        self._data = {
            uid: {
                "wav_path": uid2wavpath[uid],
                "label": self._build_label(uid),
            }
            for uid in uid2split.keys()
        }

    @property
    def all_data(self):
        return self._data

    @property
    def data_split_ids(self):
        return (
            self._split2uids["train"],
            self._split2uids["valid"],
            self._split2uids["test"],
        )

    @staticmethod
    def _get_standard_usage(dataset_root: Path):
        split_filename = SPLIT_FILE_URL.split("/")[-1]
        split_filepath = Path(_cache_root) / split_filename
        if not split_filepath.is_file():
            with FileLock(str(split_filepath) + ".lock"):
                os.system(f"wget {SPLIT_FILE_URL} -O {str(split_filepath)}")
        standard_usage = [
            line.strip().split(" ") for line in open(split_filepath, "r").readlines()
        ]

        def code2split(code: int):
            splits = ["train", "valid", "test"]
            return splits[code - 1]

        standard_usage = {uid: code2split(int(split)) for split, uid in standard_usage}
        return standard_usage

    @staticmethod
    @cache()
    def _find_wavs_with_uids(dataset_root, uids, n_jobs=4):
        def find_wav_with_uid(uid):
            found_wavs = list(dataset_root.glob(f"*/wav/{uid}"))
            assert len(found_wavs) == 1
            return uid, found_wavs[0]

        uids_with_wavs = Parallel(n_jobs=n_jobs)(
            delayed(find_wav_with_uid)(uid) for uid in tqdm(uids, desc="Search wavs")
        )
        uids2wav = {uid: wav for uid, wav in uids_with_wavs}
        return uids2wav

    @staticmethod
    def _build_label(uid):
        id_string = uid.split("/")[0]
        label = f"speaker_{int(id_string[2:]) - 10001}"
        return label


@registry.put()
def voxceleb1_for_utt_classification(dataset_root: str, n_jobs: int = 4):
    corpus = VoxCeleb1SID(dataset_root, n_jobs)
    train_data, valid_data, test_data = corpus.data_split
    return Output(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )


@registry.put()
def mini_voxceleb1(dataset_root: str, force_download=False):
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir() or force_download:
        dataset_root.mkdir(exist_ok=True, parents=True)
        os.system(f"rm -rf {dataset_root}")
        os.system(f"git lfs install")
        os.system(
            f"git clone https://huggingface.co/datasets/s3prl/mini_voxceleb1 {dataset_root}"
        )

    def prepare_datadict(split_root: str):
        files = find_files(Path(split_root))
        data = {}
        for file in files:
            file = Path(file)
            data[file.stem] = dict(
                wav_path=file.resolve(), label=file.name.split("-")[0]
            )
        return data

    dataset_root = Path(dataset_root)
    return Container(
        train_data=prepare_datadict(dataset_root / "train"),
        valid_data=prepare_datadict(dataset_root / "valid"),
        test_data=prepare_datadict(dataset_root / "test"),
    )
