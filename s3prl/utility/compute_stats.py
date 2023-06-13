import argparse
from tqdm import tqdm

import torch
import torchaudio
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("file_list")
parser.add_argument("--n_jobs", type=int, default=8)
args = parser.parse_args()


def get_audio_info(filepath: str):
    torchaudio.set_audio_backend("sox_io")
    info = torchaudio.info(filepath)
    return info


with open(args.file_list) as f:
    filepaths = [filepath.strip() for filepath in f.readlines()]

infos = Parallel(n_jobs=args.n_jobs)(
    delayed(get_audio_info)(str(f)) for f in tqdm(filepaths)
)
num_frames = [info.num_frames for info in infos]
sample_rates = [info.sample_rate for info in infos]
secs = [f / s for f, s in zip(num_frames, sample_rates)]
hours = sum(secs) / 60 / 60

print(f"Num: {len(secs)}")
print(f"Avg Sec: {torch.FloatTensor(secs).mean().item()}")
print(f"Max Sec: {torch.FloatTensor(secs).max().item()}")
print(f"Min Sec: {torch.FloatTensor(secs).min().item()}")
print(f"Std Sec: {torch.FloatTensor(secs).std().item()}")
print(f"Hour: {hours}")
