from tqdm import tqdm, trange
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset


class SnipsDataset(Dataset):
    def __init__(self, split, tokenizer, bucket_size, path, num_workers=12, ascending=False, **kwargs):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.speaker_list = kwargs[f'{split}_speakers'] if type(split) == str else kwargs[f'{split[0]}_speakers']

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
            #for spk in self.speaker_list:
            #    print('- '+spk)
            #    for wav_file in tmp_split_list:
            #        if spk in str(wav_file):
            #            split_list.append(wav_file)
            new_list = []
            uf = 0
            for i in trange(len(split_list), desc='checking files'):
                uid = str(split_list[i]).split('/')[-1].split('.wav', 1)[0].split('/')[-1]
                if uid in transcripts:
                    for spk in self.speaker_list:
                        if uid[:len(spk)] == spk:
                            new_list.append(split_list[i])
                            break
                else:
                    print(split_list[i], "Not Found")
                    uf += 1
            print("%d wav file with label not found in text file!"%uf)
            split_list = new_list
            print(f'loaded audio from {len(self.speaker_list)} speakers {str(self.speaker_list)} with {len(split_list)} examples.')
            assert len(split_list) > 0, "No data found @ {}".format(join(path,s))
            file_list += split_list
        # Read text
        #text = Parallel(n_jobs=num_workers)(
        #    delayed(read_text)(str(f)) for f in file_list)
        #text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
        text = [transcripts[str(f).split('.wav', 1)[0].split('/')[-1]] for f in file_list]
        text = [tokenizer.encode(txt) for txt in tqdm(text, desc='tokenizing')]

        # Sort dataset by text length
        #file_len = Parallel(n_jobs=num_workers)(delayed(getsize)(f) for f in file_list)
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
