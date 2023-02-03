import argparse
from pathlib import Path
from librosa.util import find_files


parser = argparse.ArgumentParser()
parser.add_argument("audio_root")
parser.add_argument("output_file")
args = parser.parse_args()

audio_root: Path = Path(args.audio_root)
assert audio_root.is_dir()

audios = find_files(audio_root)
output_file: Path = Path(args.output_file)
output_file.parent.mkdir(exist_ok=True, parents=True)

with output_file.open("w") as f:
    lines = [str(Path(audio).resolve()) + "\n" for audio in audios]
    f.writelines(lines)
