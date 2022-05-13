from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os 
import pandas as pd

'''preprocess file'''
base_path = '/home/daniel094144/data/atis'
splits = ['train', 'dev', 'test'] 
d = {}
i = 0

for split in splits:
    df = pd.read_csv(os.path.join(base_path, f"nlu_iob/iob.{split}"), sep='\t', header=None)
    text = df[0].values
    label = df[1].values

    sv_all = []
    ids = []
    with open(os.path.join(base_path, f'value_BI_{split}.txt'), 'w') as f: 
        for i, (t, l) in enumerate(zip(text, label)):
            idx = t.split()[0]
            t = t.split()[2:-1]
            l = l.split()[1:-1]

            s, v = [], []
            
            for slot, txt in zip(l, t): 
                if slot != 'O':
                    s.append(slot)
                    v.append(txt)
                    # add new slot
                    if slot not in d: 
                        d[slot] = i
                        i += 1
                  
            sv = []
            f.write(' '.join(v))
            f.write('\n')

            for slot, value in zip(s, v):
                # B-slot vegas
                sv.append(value)
                sv.append(slot)
                
            ids.append(idx)
            sv_all.append(' '.join(sv))
        
    sv_df = pd.DataFrame({'id': ids, 'label': sv_all})
    sv_df.to_csv(os.path.join(base_path, f'sv_BI_{split}.csv'))

print(list(d.keys()))


# '''tokenizer'''
output_tokenizer_path = os.path.join(base_path, "BI_tokenizer.json")

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()



trainer = BpeTrainer(special_tokens=["<PAD>", "<EOS>", '<BOS>']+list(d.keys()), vocab_size=600)
tokenizer.train(files=[os.path.join(base_path, "value_BI_train.txt"), 
                        os.path.join(base_path, "value_BI_dev.txt"), 
                        os.path.join(base_path, "value_BI_test.txt")], 
                        trainer=trainer)

tokenizer.save(output_tokenizer_path)
tokenizer_re = Tokenizer.from_file(output_tokenizer_path)

