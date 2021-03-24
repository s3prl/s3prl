import os
import random
import tqdm
import argparse
import pathlib
import shutil
from pathlib import Path
from librosa.util import find_files
from tqdm import trange

def collect_speaker_ids(roots, speaker_num):
    
    all_speaker=[ ]

    all_speaker.extend([f.path for f in os.scandir(roots) if f.is_dir()])
    
    ids = [[speaker.split("/")[-3],speaker.split("/")[-1]] for speaker in all_speaker]

    vox1 = []
    for id in ids:
        if id[0] == roots.split("/")[-2]:
            vox1.append(id[1])


    dev_speaker = random.sample(vox1, k=speaker_num)
    vox1_train = [ids for ids in vox1 if ids not in dev_speaker]
    
    train_speaker = []

    train_speaker.extend(vox1_train)

    return train_speaker, dev_speaker

def construct_dev_speaker_id_txt(dev_speakers,dev_txt_name):
    f = open(dev_txt_name, "w")
    for dev in dev_speakers:
        f.write(dev)
        f.write("\n")
    f.close()
    return


def sample_wavs_and_dump_txt(root,dev_ids, numbers, meta_data_name):
    
    wav_list = []
    count_positive = 0
    print(f"generate {numbers} sample pairs")
    for _ in trange(numbers):
        prob = random.random()
        if (prob > 0.5):
            dev_id_pair = random.sample(dev_ids, 2)

            # sample 2 wavs from different speaker
            sample1 = "/".join(random.choice(find_files(os.path.join(root,dev_id_pair[0]))).split("/")[-3:])
            sample2 = "/".join(random.choice(find_files(os.path.join(root,dev_id_pair[1]))).split("/")[-3:])

            label = "0"

            wav_list.append(" ".join([label, sample1, sample2]))
            
        else:
            dev_id_pair = random.sample(dev_ids, 1)
            
            # sample 2 wavs from same speaker
            sample1 = "/".join(random.choice(find_files(os.path.join(root,dev_id_pair[0]))).split("/")[-3:])
            sample2 = "/".join(random.choice(find_files(os.path.join(root,dev_id_pair[0]))).split("/")[-3:])

            label = "1"
            count_positive +=1

            wav_list.append(" ".join([label, sample1, sample2]))
    print("finish, then dump file ..")
    f = open(meta_data_name,"w")
    for data in wav_list:
        f.write(data+"\n")
    f.close()

    return wav_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=19941227)
    parser.add_argument('-r', '--root', default="../../librispeech/voxceleb2/dev/wav")
    parser.add_argument('-o', "--output_dir", default="../../librispeech/dev_data")
    parser.add_argument('-n',  '--speaker_num', default=40)
    parser.add_argument('-p',  '--sample_pair', default=20000)
    args = parser.parse_args()

    random.seed(args.seed)
    train_speakers, dev_speakers = collect_speaker_ids(args.root, args.speaker_num)
    construct_dev_speaker_id_txt(dev_speakers, "./downstream/sv_voxceleb1/dev_meta_data/dev_meta_speaker_ids.txt")
    wav_list = sample_wavs_and_dump_txt(args.root, dev_speakers, args.sample_pair, "./downstream/sv_voxceleb1/dev_meta_data/dev_meta_data_voxceleb2.txt")
    
    root = Path(args.root)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    new_dir = Path(args.output_dir)
    print(f"create folder {str(new_dir)} ..")
    pathlib.Path(Path(new_dir / "dev" / "wav")).mkdir(parents=True, exist_ok=True)
    print(f'create folder {str(new_dir / "dev" / "wav" )} ..')
    print(f"copy speaker_folder from {args.root}/dev/wav to {args.output_dir}/dev/wav ..")
    for speaker_fold in tqdm.tqdm(dev_speakers):
        shutil.copytree(root / speaker_fold, new_dir / "dev"/ "wav" / speaker_fold, dirs_exist_ok=True)
