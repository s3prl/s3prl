from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os 
import pandas as pd

'''preprocess file'''
# base_path = '/home/daniel094144/data/atis'
base_path = '/home/daniel094144/data/slurp/slurp/dataset/slurp'
splits = ['train', 'devel', 'test']

for split in splits:
    df = pd.read_json(os.path.join(base_path, f"{split}.jsonl"), lines=True)
    # text = df["sentence"].values
    label = df["sentence_annotation"].values
    recordings = df["recordings"].values
    slurp_id = df["slurp_id"].values
    # action is intent
    intents = df["action"].values

    sv_all = []
    ids = []
    d = {}
    intent_d = {}
    intent_index = 0
    dic_index = 0
    with open(os.path.join(base_path, f'slurp_value_{split}.txt'), 'w') as f: 
        for i, (l, r, idx, intent) in enumerate(zip(label, recordings, slurp_id, intents)):
            file_idx = [r[i]["file"] for i in range(len(r))]

            s, v = [], []
            s_with_v = []
            temp_slot_value = ''
            record_switch = False
            for word in l:
                if word == ']':
                    record_switch = False
                    s_with_v.append(temp_slot_value)
                    temp_slot_value = ''

                if record_switch: temp_slot_value = temp_slot_value + word
                if word == '[':
                    record_switch = True

            for content in s_with_v:
                I_index = False
                content = content.split(':')
                for subword in content[1].split():
                    if not I_index:
                        slot_name = 'B-' + content[0][:-1]
                        s.append(slot_name)
                        v.append(subword)
                        I_index = True
                    else:
                        slot_name = 'I-' + content[0][:-1]
                        s.append(slot_name)
                        v.append(subword)

                    if slot_name not in d: 
                        d[slot_name] = dic_index
                        dic_index += 1

                # s.append(content[0][:-1])
                # if content[0][:-1] not in d: 
                #     d[content[0][:-1]] = dic_index
                #     dic_index += 1
                # v.append(content[1][1:])
            # print(i)
            # print(s)
            # print(v)

            sv = []
            f.write(' '.join(v))
            f.write('\n')

            if intent not in intent_d:
                intent_d[intent] = intent_index
                intent_index += 1
            # add intent label in the beginning and end 
            sv.append(intent)
            for slot, value in zip(s, v):
                sv.append(value)
                sv.append(slot)
            
            sv.append(intent)
            ids.append(file_idx[0]) # first file name
            # ids.append(idx) # slurp_id
            sv_all.append(' '.join(sv))
    # print(sv_all)
    # print(ids)
    # quit()

    sv_df = pd.DataFrame({'id': ids, 'label':sv_all})
    sv_df.to_csv(os.path.join(base_path, f'sv_BIO_{split}.csv'))

'''tokenizer'''
output_tokenizer_path = os.path.join(base_path, "BIO_tokenizer.json")

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# d = {}
# i = 0
# with open(os.path.join(base_path,'slot_vocabs.txt'), 'r') as f: 
#     for line in f: 
#         line = line.strip('\n')
#         line = line[1:-1]
#         slot = line.split('|')[-1]
#         if slot not in d and slot != '': 
#             d[line] = i
#             i += 1
print(d)
print(list(d.keys()))

trainer = BpeTrainer(special_tokens=["<PAD>", "<EOS>", '<BOS>']+list(d.keys())+list(intent_d.keys()), vocab_size=1000)
tokenizer.train(files=[os.path.join(base_path, "slurp_value_train.txt"), 
                        os.path.join(base_path, "slurp_value_devel.txt"), 
                        os.path.join(base_path, "slurp_value_test.txt")], 
                        trainer=trainer)

tokenizer.save(output_tokenizer_path)
tokenizer_re = Tokenizer.from_file(output_tokenizer_path)