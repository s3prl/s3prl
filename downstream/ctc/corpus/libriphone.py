from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize, isfile
from joblib import Parallel, delayed
from torch.utils.data import Dataset


def parse_lexicon(line, tokenizer):
    line.replace('\t', ' ')
    word, *phonemes = line.split()
    for p in phonemes:
        assert p in tokenizer._vocab2idx.keys()
    return word, phonemes


def read_text(file, word2phonemes, tokenizer):
    '''Get transcription of target wave file, 
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread'''
    src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
    idx = file.split('/')[-1].split('.')[0]

    with open(src_file, 'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                transcription = line[:-1].split(' ', 1)[1]
                phonemes = []
                for word in transcription.split():
                    phonemes += word2phonemes[word]
                return tokenizer.encode(' '.join(phonemes))


class LibriPhoneDataset(Dataset):
    def __init__(self, split, tokenizer, bucket_size, path, lexicon, ascending=False, **kwargs):
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        # create word -> phonemes mapping
        word2phonemes = {}
        for lexicon_file in Path(lexicon).rglob('*.txt'):
            with open(lexicon_file, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
                for line in lines:
                    word, phonemes = parse_lexicon(line, tokenizer)
                    word2phonemes[word] = phonemes

        # List all wave files
        file_list = []
        for s in split:
            split_list = list(Path(join(path, s)).rglob("*.flac"))
            assert len(split_list) > 0, "No data found @ {}".format(join(path,s))
            file_list += split_list
        
        text = []
        for f in tqdm(file_list, desc='word -> phonemes'):
            text.append(read_text(str(f), word2phonemes, tokenizer))

        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])
    
    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)
