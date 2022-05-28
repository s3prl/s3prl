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
intent_dict = {}
i = 0
j = 0
for split in splits:
    df = pd.read_csv(os.path.join(base_path, f"nlu_iob/iob.{split}"), sep='\t', header=None)
    text = df[0].values
    label = df[1].values

    sv_all = []
    ids = []
    
    
    with open(os.path.join(base_path, f'value_BIO_{split}.txt'), 'w') as f: 
        for i, (text, label) in enumerate(zip(text, label)):
            idx = text.split()[0]
            t = text.split()[2:-1]
            l = label.split()[1:-1]

            intent = label.split()[-1]
            if intent not in intent_dict: 
                intent_dict[intent] = j
                j += 1

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
            # add intent 
            sv.append(intent)
            for slot, value in zip(s, v):
                # B-slot vegas
                sv.append(value)
                sv.append(slot)
            sv.append(intent)
            
            ids.append(idx)
            sv_all.append(' '.join(sv))
        
    sv_df = pd.DataFrame({'id': ids, 'label': sv_all})
    sv_df.to_csv(os.path.join(base_path, f'sv_BIO_{split}.csv'))

print(list(d.keys()))
print(len(list(intent_dict.keys())))


# '''tokenizer'''
output_tokenizer_path = os.path.join(base_path, "BIO_tokenizer.json")

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()



trainer = BpeTrainer(special_tokens=["<PAD>", "<EOS>", '<BOS>']+list(d.keys())+list(intent_dict.keys()), vocab_size=600)
tokenizer.train(files=[os.path.join(base_path, "value_BIO_train.txt"), 
                        os.path.join(base_path, "value_BIO_dev.txt"), 
                        os.path.join(base_path, "value_BIO_test.txt")], 
                        trainer=trainer)

tokenizer.save(output_tokenizer_path)
tokenizer_re = Tokenizer.from_file(output_tokenizer_path)

