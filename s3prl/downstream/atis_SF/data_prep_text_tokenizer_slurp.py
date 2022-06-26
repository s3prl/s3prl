from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os 
import pandas as pd

'''preprocess file'''
base_path = '/home/daniel094144/data/slurp/slurp/dataset/slurp'
splits = ['train', 'dev', 'test'] 
for split in splits: 
    df = pd.read_json(os.path.join(base_path, f"{split}.jsonl"), lines=True)
    transcriptions = df['sentence'].values
    
    with open(os.path.join(base_path, f'{split}.txt'), 'w') as f: 
        for transcription in transcriptions: 
            f.write(transcription)
            f.write('\n')


# '''tokenizer'''
output_tokenizer_path = os.path.join(base_path, "text_subword_tokenizer.json")

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()



trainer = BpeTrainer(special_tokens=["<PAD>", "<EOS>", '<BOS>'], vocab_size=1000)
tokenizer.train(files=[os.path.join(base_path, "train.txt"), 
                        os.path.join(base_path, "dev.txt"), 
                        os.path.join(base_path, "test.txt")], 
                        trainer=trainer)

tokenizer.save(output_tokenizer_path)


