"""Modified from tensorflow_datasets.features.text.*

Reference: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text_lib
"""
import abc

BERT_FIRST_IDX = 997  # Replacing the 2 tokens right before english starts as <eos> & <unk>
BERT_LAST_IDX = 29635  # Drop rest of tokens


class _BaseTextEncoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, s):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, ids, ignore_repeat=False):
        raise NotImplementedError

    @abc.abstractproperty
    def vocab_size(self):
        raise NotImplementedError

    @abc.abstractproperty
    def token_type(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def load_from_file(cls, vocab_file):
        raise NotImplementedError

    @property
    def pad_idx(self):
        return 0

    @property
    def eos_idx(self):
        return 1

    @property
    def unk_idx(self):
        return 2

    def __repr__(self):
        return "<{} vocab_size={}>".format(type(self).__name__, self.vocab_size)


class CharacterTextEncoder(_BaseTextEncoder):
    def __init__(self, vocab_list):
        # Note that vocab_list must not contain <pad>, <eos> and <unk>
        # <pad>=0, <eos>=1, <unk>=2
        self._vocab_list = ["<pad>", "<eos>", "<unk>"] + vocab_list
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}

    def encode(self, s):
        # Always strip trailing space, \r and \n
        s = s.strip("\r\n ")
        # Manually append eos to the end
        return [self.vocab_to_idx(v) for v in s] + [self.eos_idx]

    def decode(self, idxs, ignore_repeat=False):
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t-1]):
                continue
            elif idx == self.eos_idx:
                break
            else:
                vocabs.append(v)
        return "".join(vocabs)

    @classmethod
    def load_from_file(cls, vocab_file):
        with open(vocab_file, "r") as f:
            # Do not strip space because character based text encoder should
            # have a space token
            vocab_list = [line.strip("\r\n") for line in f]
        return cls(vocab_list)

    @property
    def vocab_size(self):
        return len(self._vocab_list)

    @property
    def token_type(self):
        return 'character'

    def vocab_to_idx(self, vocab):
        return self._vocab2idx.get(vocab, self.unk_idx)

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]

class CharacterTextSlotEncoder(_BaseTextEncoder):
    def __init__(self, vocab_list, slots):
        # Note that vocab_list must not contain <pad>, <eos> and <unk>
        # <pad>=0, <eos>=1, <unk>=2
        self._vocab_list = ["<pad>", "<eos>", "<unk>"] + vocab_list
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}
        self.slots = slots
        self.slot2id = {self.slots[i]:(i+len(self._vocab_list)) for i in range(len(self.slots))}
        self.id2slot = {(i+len(self._vocab_list)):self.slots[i] for i in range(len(self.slots))}


    def encode(self, s):
        # Always strip trailing space, \r and \n
        sent, iobs = s.strip('\r\n ').split('\t')
        sent = sent.split(' ')[1:-1]
        iobs = iobs.split(' ')[1:-1]
        tokens = []
        for i, (wrd, iob) in enumerate(zip(sent, iobs)):
            if wrd in "?!.,;-":
                continue
            if wrd == '&':
                wrd = 'AND'
            if iob != 'O' and (i == 0 or iobs[i-1] != iob):
                tokens.append(self.slot2id['B-'+iob])
            tokens += [self.vocab_to_idx(v) for v in wrd]
            if iob != 'O' and (i == len(sent)-1 or iobs[i+1] != iob):
                tokens.append(self.slot2id['E-'+iob])
            if i == (len(sent)-1):
                tokens.append(self.eos_idx)
            else:
                tokens.append(self.vocab_to_idx(' '))
        return tokens

    def decode(self, idxs, ignore_repeat=False):
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t-1]):
                continue
            elif idx == self.eos_idx:
                break
            else:
                vocabs.append(v)
        return "".join(vocabs)

    @classmethod
    def load_from_file(cls, vocab_file, slots_file):
        with open(vocab_file, "r") as f:
            # Do not strip space because character based text encoder should
            # have a space token
            vocab_list = [line.strip("\r\n") for line in f]
        org_slots = open(slots_file).read().split('\n')
        slots = []
        for slot in org_slots[1:]:
            slots.append('B-'+slot)
            slots.append('E-'+slot)
        return cls(vocab_list, slots)

    @property
    def vocab_size(self):
        return len(self._vocab_list) + len(self.slots)

    @property
    def token_type(self):
        return 'character-slot'

    def vocab_to_idx(self, vocab):
        return self._vocab2idx.get(vocab, self.unk_idx)

    def idx_to_vocab(self, idx):
        idx = int(idx)
        if idx < len(self._vocab_list):
            return self._vocab_list[idx]
        else:
            token = self.id2slot[idx]
            if token[0] == 'B':
                return token + ' '
            elif token[0] == 'E':
                return ' ' + token
            else:
                raise ValueError('id2slot get:', token)



class SubwordTextEncoder(_BaseTextEncoder):
    def __init__(self, spm):
        if spm.pad_id() != 0 or spm.eos_id() != 1 or spm.unk_id() != 2:
            raise ValueError(
                "Please train sentencepiece model with following argument:\n"
                "--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 --model_type=bpe --eos_piece=<eos>")
        self.spm = spm

    def encode(self, s):
        return self.spm.encode_as_ids(s)

    def decode(self, idxs, ignore_repeat=False):
        crop_idx = []
        for t, idx in enumerate(idxs):
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t-1]):
                continue
            else:
                crop_idx.append(idx)
        return self.spm.decode_ids(crop_idx)

    @classmethod
    def load_from_file(cls, filepath):
        import sentencepiece as splib
        spm = splib.SentencePieceProcessor()
        spm.load(filepath)
        spm.set_encode_extra_options(":eos")
        return cls(spm)

    @property
    def vocab_size(self):
        return len(self.spm)

    @property
    def token_type(self):
        return 'subword'


class SubwordTextSlotEncoder(_BaseTextEncoder):
    def __init__(self, spm, slots):
        if spm.pad_id() != 0 or spm.eos_id() != 1 or spm.unk_id() != 2:
            raise ValueError(
                "Please train sentencepiece model with following argument:\n"
                "--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 --model_type=bpe --eos_piece=<eos>")
        self.spm = spm
        self.slots = slots
        self.slot2id = {self.slots[i]:(i+len(self.spm)) for i in range(len(self.slots))}
        self.id2slot = {(i+len(self.spm)):self.slots[i] for i in range(len(self.slots))}

    def encode(self, s):
        sent, iobs = s.strip().split('\t')
        sent = sent.split(' ')[1:-1]
        iobs = iobs.split(' ')[1:-1]
        tokens = []
        for i, (wrd, iob) in enumerate(zip(sent, iobs)):
            if wrd in "?!.,;-":
                continue
            if wrd == '&':
                wrd = 'AND'
            if iob != 'O' and (i == 0 or iobs[i-1] != iob):
                tokens.append(self.slot2id['B-'+iob])
            tokens += self.spm.encode_as_ids(wrd)[:-1] #if i != len(sent)-1 else self.spm.encode_as_ids(wrd)
            if iob != 'O' and (i == len(sent)-1 or iobs[i+1] != iob):
                tokens.append(self.slot2id['E-'+iob])
        if tokens[-1] != 1:
            tokens.append(1)
        return tokens #self.spm.encode_as_ids(s)

    def decode(self, idxs, ignore_repeat=False):
        crop_idx = []
        for t, idx in enumerate(idxs):
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t-1]):
                continue
            else:
                crop_idx.append(idx)
        sent = []
        ret = []
        for i, x in enumerate(crop_idx):
            if x >= len(self.spm):
                ret.append(self.spm.decode_ids(sent) + [self.id2slot[x]])
            else:
                sent.append(x)
        return ret

    @classmethod
    def load_from_file(cls, filepath, slots_file):
        import sentencepiece as splib
        spm = splib.SentencePieceProcessor()
        spm.load(filepath)
        spm.set_encode_extra_options(":eos")
        org_slots = open(slots_file).read().split('\n')
        slots = []
        for slot in org_slots[1:]:
            slots.append('B-'+slot)
            slots.append('E-'+slot)
        return cls(spm, slots)

    @property
    def vocab_size(self):
        return len(self.spm) + len(self.slots)

    @property
    def token_type(self):
        return 'subword-slot'



class WordTextEncoder(CharacterTextEncoder):
    def encode(self, s):
        # Always strip trailing space, \r and \n
        s = s.strip("\r\n ")
        # Space as the delimiter between words
        words = s.split(" ")
        # Manually append eos to the end
        return [self.vocab_to_idx(v) for v in words] + [self.eos_idx]

    def decode(self, idxs, ignore_repeat=False):
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t-1]):
                continue
            else:
                vocabs.append(v)
        return " ".join(vocabs)

    @property
    def token_type(self):
        return 'word'


class BertTextEncoder(_BaseTextEncoder):
    """Bert Tokenizer.

    https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_bert.py
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tokenizer.pad_token = "<pad>"
        self._tokenizer.eos_token = "<eos>"
        self._tokenizer.unk_token = "<unk>"

    def encode(self, s):
        # Reduce vocab size manually
        reduced_idx = []
        for idx in self._tokenizer.encode(s):
            try:
                r_idx = idx-BERT_FIRST_IDX
                assert r_idx > 0
                reduced_idx.append(r_idx)
            except:
                reduced_idx.append(self.unk_idx)
        reduced_idx.append(self.eos_idx)
        return reduced_idx

    def decode(self, idxs, ignore_repeat=False):
        crop_idx = []
        for t, idx in enumerate(idxs):
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t-1]):
                continue
            else:
                # Shift to correct idx for bert tokenizer
                crop_idx.append(idx+BERT_FIRST_IDX)
        return self._tokenizer.decode(crop_idx)

    @property
    def vocab_size(self):
        return BERT_LAST_IDX-BERT_FIRST_IDX+1

    @property
    def token_type(self):
        return "bert"

    @classmethod
    def load_from_file(cls, vocab_file):
        from pytorch_transformers import BertTokenizer
        return cls(BertTokenizer.from_pretrained(vocab_file))

    @property
    def pad_idx(self):
        return 0

    @property
    def eos_idx(self):
        return 1

    @property
    def unk_idx(self):
        return 2


def load_text_encoder(mode, vocab_file, slots_file=None):
    if mode == "character":
        return CharacterTextEncoder.load_from_file(vocab_file)
    elif mode == "character-slot":
        return CharacterTextSlotEncoder.load_from_file(vocab_file, slots_file)
    elif mode == "subword":
        return SubwordTextEncoder.load_from_file(vocab_file)
    elif mode == "subword-slot":
        return SubwordTextSlotEncoder.load_from_file(vocab_file, slots_file)
    elif mode == "word":
        return WordTextEncoder.load_from_file(vocab_file)
    elif mode.startswith("bert-"):
        return BertTextEncoder.load_from_file(mode)
    else:
        raise NotImplementedError("`{}` is not yet supported.".format(mode))
