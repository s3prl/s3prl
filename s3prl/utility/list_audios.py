import argparse
from pathlib import Path

import torchaudio
from tqdm import tqdm
from librosa.util import find_files
from joblib import Parallel, delayed


def get_audio_info(audio_path: str):
    torchaudio.set_audio_backend("sox_io")
    info = torchaudio.info(audio_path)
    num_frames = info.num_frames
    sample_rate = info.sample_rate
    return num_frames, sample_rate


def get_filename(audio_path: str, last_n_parts: int):
    audio_path: Path = Path(audio_path)
    audio_path_parts = audio_path.parts
    filename = list(audio_path_parts[-last_n_parts:])
    filename[-1] = Path(filename[-1]).stem
    return "-".join(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_root")
    parser.add_argument("output_path")
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument(
        "--last_n_parts",
        type=int,
        default=1,
        help="last n parts in the filepath to use as filename",
    )
    args = parser.parse_args()

    file_paths = find_files(args.audio_root)
    file_paths = sorted(file_paths)
    file_paths = [str(Path(path).absolute()) for path in file_paths]
    file_names = [
        get_filename(file_path, args.last_n_parts) for file_path in file_paths
    ]
    assert len(set(file_names)) == len(
        file_names
    ), "Duplicated filenames are not allowed"

    file_infos = Parallel(n_jobs=args.n_jobs)(
        delayed(get_audio_info)(path) for path in tqdm(file_paths)
    )
    all_num_frames, sample_rates = zip(*file_infos)

    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output_path, "w") as f:
        for name, path, num_frames, sample_rate in zip(
            file_names, file_paths, all_num_frames, sample_rates
        ):
            print(name, path, num_frames, sample_rate, sep="\t", file=f)
