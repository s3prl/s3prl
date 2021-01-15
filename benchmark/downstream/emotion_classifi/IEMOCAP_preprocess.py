import os
from os.path import basename, splitext, join as path_join
import re
import random
import argparse
import json
from librosa.util import find_files

LABEL_DIR_PATH = 'dialog/EmoEvaluation'
WAV_DIR_PATH = 'sentences/wav'


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    return vars(parser.parse_args())


def get_wav_paths(data_dirs):
    wav_paths = find_files(data_dirs)
    wav_dict = {}
    for wav_path in wav_paths:
        wav_name = splitext(basename(wav_path))[0]
        start = wav_path.find('Session')
        wav_path = wav_path[start:]
        wav_dict[wav_name] = wav_path

    return wav_dict


def preprocess(data_dirs, paths, out_path):
    meta_data = []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            with open(path_join(label_dir, label_path)) as f:
                for line in f:
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[2] not in ['neu', 'hap', 'ang', 'sad', 'exc']:
                        continue
                    if line[1] not in wav_paths:
                        continue
                    meta_data.append({
                        'path': wav_paths[line[1]],
                        'label': line[2].replace('exc', 'hap'),
                        'speaker': re.split('_', basename(wav_paths[line[1]]))[0]
                    })
    data = {
        'labels': {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3},
        'meta_data': meta_data
    }
    with open(out_path, 'w') as f:
        json.dump(data, f)


def main(data_dir, out_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    random.shuffle(paths)
    os.makedirs(out_dir, exist_ok=True)
    print(paths)
    preprocess(data_dir, paths[:4], path_join(out_dir, 'train_meta_data.json'))
    preprocess(data_dir, [paths[4]], path_join(out_dir, 'test_meta_data.json'))


if __name__ == "__main__":
    main(**parse_args())
