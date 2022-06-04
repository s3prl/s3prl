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
    
    
    with open(os.path.join(base_path, f'value_split_{split}.txt'), 'w') as f: 
        for i, (text, label) in enumerate(zip(text, label)):
            idx = text.split()[0]
            t = text.split()[2:-1]
            l = label.split()[1:-1]

            intent = label.split()[-1]

            s, v = [], []
            skip = False
            for slot, txt in zip(l, t): 
                if slot != 'O':
                    if slot.split('-')[0] == 'B':
                        slt = slot.split('-')[-1]
                        
                        # preprocess slot 
                        if len(slt.split('.')) > 1: 
                            general_slt = slt.split('.')[-1]
                            specific_slt = slt.split('.')[0]
                            if len(specific_slt.split('_')) > 1:
                                specific_slt = specific_slt.split('_')[0]
                            slt = [general_slt, specific_slt]

                        s.append(slt)
                        v.append(txt)
                        # add new slot
                        # split slot
                        if type(slt) == list:
                            for x in slt: 
                                if x not in d: 
                                    d[x] = i
                                    i += 1
                            
                        # simple slot
                        else: 
                            if slt not in d: 
                                d[slt] = i
                                i += 1

                    if slot.split('-')[0] == 'I':            
                        try: 
                            v[-1] += f' {txt}'
                        except: 
                            print(idx)
                            skip = True

            if not skip or len(v) == 0:           
                sv = []
                f.write(' '.join(v))
                f.write('\n')
                
                # add intent to dict
                intent = intent.split('#')[0]
                if intent not in intent_dict: 
                    intent_dict[intent] = i

                sv.append(intent)
                sv.append(';')
                for slot, value in zip(s, v):
                    sv.append(value)
                    sv.append(':')
                    if type(slot) == list:
                        for x in slot:
                            sv.append(x)
                    else: 
                        sv.append(slot)
                    
                    sv.append(';')

                sv.append(intent)

                ids.append(idx)
                sv_all.append(' '.join(sv))
        
    sv_df = pd.DataFrame({'id': ids, 'label': sv_all})
    sv_df.to_csv(os.path.join(base_path, f'sv_split_{split}.csv'))

print(list(d.keys()))
print(len(list(intent_dict.keys())))


# '''tokenizer'''
output_tokenizer_path = os.path.join(base_path, "split_tokenizer.json")

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()



trainer = BpeTrainer(special_tokens=["<PAD>", "<EOS>", '<BOS>', ';', ':']+list(d.keys())+list(intent_dict.keys()), vocab_size=600)
tokenizer.train(files=[os.path.join(base_path, "value_split_train.txt"), 
                        os.path.join(base_path, "value_split_dev.txt"), 
                        os.path.join(base_path, "value_split_test.txt")], 
                        trainer=trainer)

tokenizer.save(output_tokenizer_path)
tokenizer_re = Tokenizer.from_file(output_tokenizer_path)

