import fairseq
from fairseq.data import Dictionary, encoders
import csv
from argparse import Namespace
import torch

class AdditionalDataset:

    @classmethod
    def from_tsv(cls, file, key, bpe_tokenizer=None, pre_tokenizer=None):

        data = []
        with open(file, 'r') as file:
            reader = csv.DictReader(file,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            for line in reader:
                data.append(line[key])
        
        return cls(data, bpe_tokenizer, pre_tokenizer)

    def __init__(self, data, dictionary, bpe_tokenizer=None, pre_tokenizer=None):

        self.data = data
        self.bpe_tokenizer = bpe_tokenizer
        self.pre_tokenizer = pre_tokenizer
        self.dictionary = dictionary

    def _create_target(self, index):
        
        tokenized = self._tokenize_text(self.data[index])
        target = self.dictionary.encode_line(
            tokenized, add_if_not_exist=False, append_eos=True
        ).long()
        
        return target

    def get_addtional_input(self, id_list):

        target = [self._create_target(id) for id in id_list]

        batched_target = fairseq.data.data_utils.collate_tokens(
            target,
            self.dictionary.pad(),
            self.dictionary.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        target_lengths = torch.tensor(
            [t.size(0) for t in target], dtype=torch.long
        )
        prev_output_tokens = fairseq.data.data_utils.collate_tokens(
            target,
            self.dictionary.pad(),
            self.dictionary.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )

        ntokens = sum(t.size(0) for t in target)

        return {
            "target": batched_target,
            "prev_output_tokens": prev_output_tokens,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
        }
        
    def _tokenize_text(self, text):

        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)

        return text
