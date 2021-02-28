from tqdm import tqdm, trange
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset
# Additional (official) text src provided
OFFICIAL_TXT_SRC = ['librispeech-lm-norm.txt']
# Remove longest N sentence in librispeech-lm-norm.txt
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading LibriSpeech
READ_FILE_THREADS = 4

# transcripts_file = open('/groups/yungsung/End-to-end-ASR-Pytorch/data/SNIPS/all-trans.txt').readlines()
# transcripts = {}
# for line in transcripts_file:
#     line = line.strip().split(' ')
#     index = line[0]
#     sent = ' '.join(line[1:])
#     transcripts[index] = sent

# def read_text(file):
#     '''Get transcription of target wave file, 
#        it's somewhat redundant for accessing each txt multiplt times,
#        but it works fine with multi-thread'''
#     #src_file = file.rsplit('/', 2)[0]+'/bopomo.trans.txt'
#     idx = file.split('/')[-1].split('.')[0]
#     return transcripts[idx]
# 
#     with open(src_file, 'r') as fp:
#         for line in fp:
#             if idx == line.split(' ')[0]:
#                 return line[:-1].split(' ', 1)[1]


class SnipsDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        # Load transcription
        transcripts_file = open(join(self.path, 'all.iob.snips.txt' if '-slot' in tokenizer.token_type else 'all-trans.txt')).readlines()
        transcripts = {}
        for line in transcripts_file:
            line = line.strip().split(' ')
            index = line[0]
            sent = ' '.join(line[1:])
            transcripts[index] = sent

        # List all wave files
        file_list = []
        for s in split:
            split_list = list(Path(join(path, s)).rglob("*.wav"))
            new_list = []
            uf = 0
            for i in trange(len(split_list), desc='checking files'):
                if str(split_list[i]).split('/')[-1].split('.wav', 1)[0].split('/')[-1] in transcripts:
                    new_list.append(split_list[i])
                else:
                    print(split_list[i], "Not Found")
                    uf += 1
            print("%d wav file with label not found in text file!"%uf)
            split_list = new_list
            assert len(split_list) > 0, "No data found @ {}".format(join(path,s))
            file_list += split_list
        # Read text
        #text = Parallel(n_jobs=READ_FILE_THREADS)(
        #    delayed(read_text)(str(f)) for f in file_list)
        #text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
        text = [transcripts[str(f).split('.wav', 1)[0].split('/')[-1]] for f in file_list]
        text = [tokenizer.encode(txt) for txt in tqdm(text, desc='tokenizing')]

        # Sort dataset by text length
        #file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)


class SnipsTextDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.encode_on_fly = False
        read_txt_src = []

        # Load transcription
        transcripts_file = open(join(self.path, 'all.iob.snips.txt' if '-slot' in tokenizer.token_type else 'all-trans.txt')).readlines()
        transcripts = {}
        for line in transcripts_file:
            line = line.strip().split(' ')
            index = line[0]
            sent = ' '.join(line[1:])
            transcripts[index] = sent

        # List all wave files
        file_list, all_sent = [], []

        for s in split:
            if s in OFFICIAL_TXT_SRC:
                self.encode_on_fly = True
                with open(join(path, s), 'r') as f:
                    all_sent += f.readlines()
            file_list += list(Path(join(path, s)).rglob("*.wav"))
        assert (len(file_list) > 0) or (len(all_sent)
                                        > 0), "No data found @ {}".format(path)

        # Read text
        #text = Parallel(n_jobs=READ_FILE_THREADS)(
        #    delayed(read_text)(str(f)) for f in file_list)
        text = [transcripts[str(f).split('.wav', 1)[0].split('/')[-1]] for f in file_list]
        all_sent.extend(text)
        del text

        # Encode text
        if self.encode_on_fly:
            self.tokenizer = tokenizer
            self.text = all_sent
        else:
            self.text = [tokenizer.encode(txt) for txt in tqdm(all_sent)]
        del all_sent

        # Read file size and sort dataset by file size (Note: feature len. may be different)
        self.text = sorted(self.text, reverse=True, key=lambda x: len(x))
        if self.encode_on_fly:
            del self.text[:REMOVE_TOP_N_TXT]

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.text)-self.bucket_size, index)
            if self.encode_on_fly:
                for i in range(index, index+self.bucket_size):
                    if type(self.text[i]) is str:
                        self.text[i] = self.tokenizer.encode(self.text[i])
            # Return a bucket
            return self.text[index:index+self.bucket_size]
        else:
            if self.encode_on_fly and type(self.text[index]) is str:
                self.text[index] = self.tokenizer.encode(self.text[index])
            return self.text[index]

    def __len__(self):
        return len(self.text)
