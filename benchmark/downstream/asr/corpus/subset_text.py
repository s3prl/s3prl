from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed

path = '/Home/daniel094144/LAS/LibriSpeech'
split =  ['train-clean-100']

def read_text(f, file):
    #idx = file.split('/')[-1].split('.')[0]
    
    with open(file, 'r') as fp:
        for line in fp:
            print(line[:-1].split(' ', 1)[1])
            f.write(line[:-1].split(' ', 1)[1])
            f.write("\n")


f = open("clean100.txt", "a")
file_list = []
for s in split:
    split_list = list(Path(join(path, s)).rglob("*.trans.txt"))
    for txt_file in split_list:
        print(txt_file)
        read_text(f, txt_file)
f.close()