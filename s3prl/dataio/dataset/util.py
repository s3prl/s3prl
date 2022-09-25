import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List

from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_info(dataset, *names: List[str], n_jobs: int = 6, cache_dir: str = None):
    logger.info(
        f"Getting info from dataset {dataset.__class__.__qualname__}: {' '.join(names)}"
    )
    if isinstance(cache_dir, (str, Path)):
        logger.info(f"Using cached info in {cache_dir}")
        cache_dir: Path = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = dataset.getinfo(0)
        for name in names:
            assert name in data
    except:
        fn = dataset.__getitem__
    else:
        fn = dataset.getinfo

    def _get(idx):
        if isinstance(cache_dir, (str, Path)):
            cache_path: Path = Path(cache_dir) / f"{idx}.json"
            if cache_path.is_file():
                with cache_path.open() as f:
                    cached = json.load(f)
                    all_presented = True
                    for name in names:
                        if name not in cached:
                            all_presented = False
                    if all_presented:
                        return cached

        data = fn(idx)

        info = {}
        for name in names:
            info[name] = data[name]

        if isinstance(cache_dir, (str, Path)):
            cache_path: Path = Path(cache_dir) / f"{idx}.json"
            with cache_path.open("w") as f:
                json.dump(info, f)

        return info

    infos = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_get)(idx) for idx in tqdm(range(len(dataset)))
    )

    organized_info = defaultdict(list)
    for info in infos:
        for k, v in info.items():
            organized_info[k].append(v)

    output = []
    for name in names:
        output.append(organized_info[name])

    if len(output) == 1:
        return output[0]
    else:
        return output
