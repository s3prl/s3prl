from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os 
import pandas as pd

'''preprocess file'''
base_path = '/home/daniel094144/data/atis'
splits = ['train', 'dev', 'test'] 

for split in splits:
    df = pd.read_csv(os.path.join(base_path, f"nlu_iob/iob.{split}"), sep='\t', header=None)
    text = df[0].values
    label = df[1].values

    sv_all = []
    ids = []
    with open(os.path.join(base_path, f'value_{split}.txt'), 'w') as f: 
        for i, (t, l) in enumerate(zip(text, label)):
            idx = t.split()[0]
            t = t.split()[2:-1]
            l = l.split()[1:-1]

            s, v = [], []
            skip = False
            for slot, txt in zip(l, t): 
                if slot != 'O':
                    if slot.split('-')[0] == 'B':
                        s.append(slot.split('-')[-1])
                        v.append(txt)
                    if slot.split('-')[0] == 'I':            
                        try: 
                            v[-1] += f' {txt}'
                        except: 
                            print(i)
                            skip = True
            if not skip or len(v) == 0:           
                sv = []
                f.write(' '.join(v))
                f.write('\n')

                for slot, value in zip(s, v):
                    sv.append(slot)
                    sv.append(value)
                ids.append(idx)
                sv_all.append(' '.join(sv))
        
    sv_df = pd.DataFrame({'id': ids, 'label':sv_all})
    sv_df.to_csv(os.path.join(base_path, f'sv_{split}.csv'))

'''tokenizer'''
output_tokenizer_path = os.path.join(base_path, "tokenizer.json")

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

d = {}
i = 0
with open(os.path.join(base_path,'slot_vocabs.txt'), 'r') as f: 
    for line in f: 
        line = line.strip('\n')
        line = line[1:-1]
        slot = line.split('|')[-1]
        if slot not in d and slot != '': 
            d[line] = i
            i += 1
print(d)
print(list(d.keys()))

trainer = BpeTrainer(special_tokens=["<PAD>", "<EOS>", '<BOS>']+list(d.keys()), vocab_size=600)
tokenizer.train(files=[os.path.join(base_path, "value_train.txt"), 
                        os.path.join(base_path, "value_dev.txt"), 
                        os.path.join(base_path, "value_test.txt")], 
                        trainer=trainer)

tokenizer.save(output_tokenizer_path)
tokenizer_re = Tokenizer.from_file(output_tokenizer_path)

