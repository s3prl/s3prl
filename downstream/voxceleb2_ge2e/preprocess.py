import os
from librosa.util import find_files
import random

roots = {"Voxceleb1":"/home/pohan/data/librispeech/vox1_train_verifi/wav", "Voxceleb2":"/home/pohan/data/librispeech/vox2_dev/wav"}

def collect_speaker_ids(roots, speaker_num):
    
    all_speaker=[ ]
    for key in list(roots.keys()):
        all_speaker.extend([f.path for f in os.scandir(roots[key]) if f.is_dir()])
    
    ids = [[speaker.split("/")[-3],speaker.split("/")[-1]] for speaker in all_speaker]

    vox1 = []
    vox2 = []
    
    for id in ids:
        if id[0] == roots["Voxceleb1"].split("/")[-2]:
            vox1.append(id[1])
        if id[0] == roots["Voxceleb2"].split("/")[-2]:
            vox2.append(id[1])

    dev_speaker = random.sample(vox1, k=speaker_num)
    vox1_train = [ids for ids in vox1 if ids not in dev_speaker]
    
    train_speaker = []

    train_speaker.extend(vox1_train)
    train_speaker.extend(vox2)

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
    for _ in range(numbers):
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
    
    f = open(meta_data_name,"w")
    for data in wav_list:
        f.write(data+"\n")
    f.close()

    return wav_list


if __name__ == "__main__":
    train_speakers, dev_speakers = collect_speaker_ids(roots, 51)
    construct_dev_speaker_id_txt(dev_speakers, "dev_meta_data/dev_speaker_ids.txt")
    wav_list = sample_wavs_and_dump_txt(roots["Voxceleb1"], dev_speakers, 4000, "dev_meta_data/dev_meta_data.txt")

