"""
Load tokenizer to encode & decode

Modified from tensorflow_datasets.features.text.*
Reference: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text_lib

Authors:
  * Heng-Jui Chang 2022
"""

import abc
import re
import tempfile
from typing import List

# Replacing the 2 tokens right before english starts as <eos> & <unk>
BERT_FIRST_IDX = 997
# Drop rest of tokens
BERT_LAST_IDX = 29635

# Default vocabularies
CHARACTER_VOCAB = list(" 'ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PHONEME_VOCAB = "SIL SPN AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW0 UW1 UW2 V W Y Z ZH".split(
    " "
)


__all__ = [
    "CharacterTokenizer",
    "CharacterSlotTokenizer",
    "SubwordTokenizer",
    "SubwordSlotTokenizer",
    "WordTokenizer",
    "PhonemeTokenizer",
    "load_tokenizer",
    "default_phoneme_tokenizer",
]


class Tokenizer:
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def encode(self, text: str, iob: str = None) -> List[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, idxs: List[int], ignore_repeat: bool = False) -> str:
        raise NotImplementedError

    def __len__(self):
        return self.vocab_size

    @abc.abstractproperty
    def vocab_size(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def token_type(self) -> str:
        raise NotImplementedError

    @abc.abstractclassmethod
    def load_from_file(cls, vocab_file: str):
        raise NotImplementedError

    @property
    def pad_idx(self) -> int:
        return 0

    @property
    def eos_idx(self) -> int:
        return 1

    @property
    def unk_idx(self) -> int:
        return 2

    def __repr__(self) -> str:
        return "<{} vocab_size={}>".format(type(self).__name__, self.vocab_size)


class CharacterTokenizer(Tokenizer):
    """Character tokenizer."""

    def __init__(self, vocab_list: List[str] = None):
        super().__init__()

        if vocab_list is None:
            vocab_list = CHARACTER_VOCAB

        for tok in ["<pad>", "<eos>", "<unk>"]:
            # Note that vocab_list must not contain <pad>, <eos> and <unk>
            assert tok not in vocab_list

        # <pad> = 0, <eos> = 1, <unk> = 2
        self._vocab_list = ["<pad>", "<eos>", "<unk>"] + vocab_list
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}

    def encode(self, s: str) -> List[int]:
        # Always strip trailing space, \r and \n
        s = s.strip("\r\n ")
        # Manually append eos to the end
        return [self.vocab_to_idx(v) for v in s] + [self.eos_idx]

    def decode(self, idxs: List[int], ignore_repeat: bool = False) -> str:
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t - 1]):
                continue
            elif idx == self.eos_idx:
                break
            else:
                vocabs.append(v)
        return "".join(vocabs)

    @classmethod
    def load_from_file(cls, vocab_file: str = None, vocab_list: List[str] = None):
        if vocab_file is not None:
            with open(vocab_file, "r") as f:
                # Do not strip space because character based text encoder should
                # have a space token
                vocab_list = [line.strip("\r\n") for line in f]
        elif vocab_list is not None:
            pass
        else:
            raise ValueError(
                "No vocabulary information give, please specify either vocab_file or vocab_list."
            )

        return cls(vocab_list)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_list)

    @property
    def token_type(self) -> str:
        return "character"

    def vocab_to_idx(self, vocab):
        return self._vocab2idx.get(vocab, self.unk_idx)

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]


class CharacterSlotTokenizer(Tokenizer):
    """Character tokenizer with slots."""

    def __init__(self, vocab_list: List[str], slots: List[str]):
        super().__init__()

        for tok in ["<pad>", "<eos>", "<unk>"]:
            # Note that vocab_list must not contain <pad>, <eos> and <unk>
            assert tok not in vocab_list

        # <pad> = 0, <eos> = 1, <unk> = 2
        self._vocab_list = ["<pad>", "<eos>", "<unk>"] + vocab_list
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}
        self.space_idx = self.vocab_to_idx(" ")
        self.slots = slots
        self.slot2id = {
            self.slots[i]: (i + len(self._vocab_list)) for i in range(len(self.slots))
        }
        self.id2slot = {
            (i + len(self._vocab_list)): self.slots[i] for i in range(len(self.slots))
        }

    def encode(self, sent: str, iobs: str) -> List[int]:
        # Always strip trailing space, \r and \n
        sent = sent.strip("\r\n ")
        iobs = iobs.strip("\r\n ")
        sent = re.sub(" +", " ", sent).strip(" ")
        sent = sent.split(" ")
        iobs = iobs.split(" ")
        assert len(sent) == len(
            iobs
        ), f"transcription and iobs should have same number of words (split by space)"

        if sent[0] == "BOS":
            sent = sent[1:]
            iobs = iobs[1:]

        if sent[-1] == "EOS":
            sent = sent[:-1]
            iobs = iobs[:-1]

        tokens = []
        for i, (wrd, iob) in enumerate(zip(sent, iobs)):
            if iob != "O" and (i == 0 or iobs[i - 1] != iob):
                tokens.append(self.slot2id["B-" + iob])
            tokens += [self.vocab_to_idx(v) for v in wrd]
            if iob != "O" and (i == len(sent) - 1 or iobs[i + 1] != iob):
                tokens.append(self.slot2id["E-" + iob])
            if i == (len(sent) - 1):
                tokens.append(self.eos_idx)
            else:
                if len(tokens) > 0 and tokens[-1] != self.space_idx:
                    tokens.append(self.space_idx)
        assert tokens[-1] == self.eos_idx
        return tokens

    def decode(self, idxs: List[int], ignore_repeat: bool = False) -> str:
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t - 1]):
                continue
            elif idx == self.eos_idx:
                break
            else:
                vocabs.append(v)
        return "".join(vocabs)

    @classmethod
    def load_from_file(cls, vocab_file: str, slots_file: str):
        with open(vocab_file, "r") as f:
            # Do not strip space because character based text encoder should
            # have a space token
            vocab_list = [line.strip("\r\n") for line in f]
        org_slots = open(slots_file).read().split("\n")
        slots = []
        for slot in [slot for slot in org_slots if slot != "O"]:
            slots.append("B-" + slot)
            slots.append("E-" + slot)
        return cls(vocab_list, slots)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_list) + len(self.slots)

    @property
    def token_type(self) -> str:
        return "character-slot"

    def vocab_to_idx(self, vocab):
        return self._vocab2idx.get(vocab, self.unk_idx)

    def idx_to_vocab(self, idx):
        idx = int(idx)
        if idx < len(self._vocab_list):
            return self._vocab_list[idx]
        else:
            token = self.id2slot[idx]
            if token[0] == "B":
                return token + " "
            elif token[0] == "E":
                return " " + token
            else:
                raise ValueError("id2slot get:", token)


class SubwordTokenizer(Tokenizer):
    """Subword tokenizer using sentencepiece."""

    def __init__(self, spm):
        super().__init__()
        if spm.pad_id() != 0 or spm.eos_id() != 1 or spm.unk_id() != 2:
            raise ValueError(
                "Please train sentencepiece model with following argument:\n"
                "--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 --model_type=bpe --eos_piece=<eos>"
            )
        self.spm = spm

    def encode(self, s: str) -> List[int]:
        tokens = self.spm.encode_as_ids(s)
        assert tokens[-1] == self.eos_idx
        return tokens

    def decode(self, idxs: List[int], ignore_repeat: bool = False) -> str:
        crop_idx = []
        for t, idx in enumerate(idxs):
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (
                ignore_repeat and t > 0 and idx == idxs[t - 1]
            ):
                continue
            else:
                crop_idx.append(idx)
        return self.spm.decode_ids(crop_idx)

    @classmethod
    def load_from_file(cls, filepath: str):
        import sentencepiece as splib

        spm = splib.SentencePieceProcessor()
        spm.load(filepath)
        spm.set_encode_extra_options("eos")
        return cls(spm)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.spm.set_encode_extra_options("eos")

    @property
    def vocab_size(self) -> int:
        return len(self.spm)

    @property
    def token_type(self) -> str:
        return "subword"


class SubwordSlotTokenizer(Tokenizer):
    """Subword tokenizer with slots."""

    def __init__(self, spm, slots):
        super().__init__()
        if spm.pad_id() != 0 or spm.eos_id() != 1 or spm.unk_id() != 2:
            raise ValueError(
                "Please train sentencepiece model with following argument:\n"
                "--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 --model_type=bpe --eos_piece=<eos>"
            )
        self.spm = spm
        self.slots = slots
        self.slot2id = {
            self.slots[i]: (i + len(self.spm)) for i in range(len(self.slots))
        }
        self.id2slot = {
            (i + len(self.spm)): self.slots[i] for i in range(len(self.slots))
        }

    def encode(self, sent: str, iobs: str) -> List[int]:
        # Always strip trailing space, \r and \n
        sent = sent.strip("\r\n ")
        iobs = iobs.strip("\r\n ")
        sent = re.sub(" +", " ", sent).strip(" ")
        sent = sent.split(" ")
        iobs = iobs.split(" ")
        assert len(sent) == len(
            iobs
        ), f"transcription and iobs should have same number of words (split by space)"

        if sent[0] == "BOS":
            sent = sent[1:]
            iobs = iobs[1:]

        if sent[-1] == "EOS":
            sent = sent[:-1]
            iobs = iobs[:-1]

        tokens = []
        for i, (wrd, iob) in enumerate(zip(sent, iobs)):
            if iob != "O" and (i == 0 or iobs[i - 1] != iob):
                tokens.append(self.slot2id["B-" + iob])
            encoded = self.spm.encode_as_ids(wrd)
            assert encoded[-1] == self.eos_idx
            tokens += encoded[:-1]  # drop eos
            if iob != "O" and (i == len(sent) - 1 or iobs[i + 1] != iob):
                tokens.append(self.slot2id["E-" + iob])
        assert tokens[-1] != self.eos_idx
        tokens.append(self.eos_idx)
        return tokens

    def decode(self, idxs: List[int], ignore_repeat: bool = False) -> str:
        crop_idx = []
        for t, idx in enumerate(idxs):
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (
                ignore_repeat and t > 0 and idx == idxs[t - 1]
            ):
                continue
            else:
                crop_idx.append(idx)

        sent, ret = [], []
        for i, x in enumerate(crop_idx):
            if x >= len(self.spm):  # x is slot token
                slot = self.id2slot[x]
                ret.append(slot)
                if len(sent) > 0:
                    decoded = self.spm.decode_ids(sent)
                    ret.insert(-1, decoded)
                    sent = []
            else:  # x is a regular token interpretable by spm
                sent.append(x)
        return " ".join(ret)

    @classmethod
    def load_from_file(cls, filepath: str, slots_file: str):
        import sentencepiece as splib

        spm = splib.SentencePieceProcessor()
        spm.load(filepath)
        spm.set_encode_extra_options(":eos")
        org_slots = open(slots_file).read().split("\n")
        slots = []
        for slot in [slot for slot in org_slots if slot != "O"]:
            slots.append("B-" + slot)
            slots.append("E-" + slot)
        return cls(spm, slots)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.spm.set_encode_extra_options("eos")

    @property
    def vocab_size(self) -> int:
        return len(self.spm) + len(self.slots)

    @property
    def token_type(self) -> str:
        return "subword-slot"


class WordTokenizer(CharacterTokenizer):
    """Word tokenizer."""

    def encode(self, s: str) -> List[int]:
        # Always strip trailing space, \r and \n
        s = s.strip("\r\n ")
        # Space as the delimiter between words
        words = s.split(" ")
        # Manually append eos to the end
        return [self.vocab_to_idx(v) for v in words] + [self.eos_idx]

    def decode(self, idxs: List[int], ignore_repeat: bool = False) -> str:
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (
                ignore_repeat and t > 0 and idx == idxs[t - 1]
            ):
                continue
            else:
                vocabs.append(v)
        return " ".join(vocabs)

    @property
    def token_type(self) -> str:
        return "word"


class PhonemeTokenizer(WordTokenizer):
    """Phoneme tokenizer."""

    @property
    def token_type(self) -> str:
        return "phoneme"


class BertTokenizer(Tokenizer):
    """Bert Tokenizer.

    https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_bert.py
    """

    def __init__(self, tokenizer):
        super().__init__()
        self._tokenizer = tokenizer
        self._tokenizer.pad_token = "<pad>"
        self._tokenizer.eos_token = "<eos>"
        self._tokenizer.unk_token = "<unk>"

    def encode(self, s: str) -> List[int]:
        # Reduce vocab size manually
        reduced_idx = []
        for idx in self._tokenizer.encode(s):
            try:
                r_idx = idx - BERT_FIRST_IDX
                assert r_idx > 0
                reduced_idx.append(r_idx)
            except AssertionError:
                reduced_idx.append(self.unk_idx)
        reduced_idx.append(self.eos_idx)
        return reduced_idx

    def decode(self, idxs: List[int], ignore_repeat: bool = False) -> str:
        crop_idx = []
        for t, idx in enumerate(idxs):
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (
                ignore_repeat and t > 0 and idx == idxs[t - 1]
            ):
                continue
            else:
                # Shift to correct idx for bert tokenizer
                crop_idx.append(idx + BERT_FIRST_IDX)
        return self._tokenizer.decode(crop_idx)

    @classmethod
    def load_from_file(cls, vocab_file: str):
        from pytorch_transformers import BertTokenizer as bert_tokenizer

        return cls(bert_tokenizer.from_pretrained(vocab_file))

    @property
    def vocab_size(self) -> int:
        return BERT_LAST_IDX - BERT_FIRST_IDX + 1

    @property
    def token_type(self) -> str:
        return "bert"


def load_tokenizer(
    mode: str,
    vocab_file: str = None,
    vocab_list: List[str] = None,
    slots_file: str = None,
) -> Tokenizer:
    """Load a text tokenizer.

    Args:
        mode (str): Mode ("character", "character-slot", "subword", "subword-slot", "word", "bert-...")
        vocab_file (str, optional): Path to vocabularies. Defaults to None.
        vocab_list (List[str], optional): List of vocabularies. Defaults to None.
        slots_file (str, optional): Path to slots. Defaults to None.

    Raises:
        NotImplementedError: If mode is not implemented.

    Returns:
        Tokenizer: Text tokenizer.
    """
    assert (
        int(vocab_file is not None) + int(vocab_list is not None) <= 1
    ), "For 'vocab_file' and 'vocab_list', at most one argument can be presented"

    with tempfile.NamedTemporaryFile("w") as f:
        if vocab_list is not None:
            f.writelines([f"{vocab}\n" for vocab in vocab_list])
            f.flush()
            vocab_file = f.name

        if slots_file is not None and not mode.endswith("slot"):
            mode = f"{mode}-slot"

        if mode == "character":
            return CharacterTokenizer.load_from_file(vocab_file)
        elif mode == "character-slot":
            return CharacterSlotTokenizer.load_from_file(vocab_file, slots_file)
        elif mode == "subword":
            return SubwordTokenizer.load_from_file(vocab_file)
        elif mode == "subword-slot":
            return SubwordSlotTokenizer.load_from_file(vocab_file, slots_file)
        elif mode == "word":
            return WordTokenizer.load_from_file(vocab_file)
        elif mode == "phoneme":
            return PhonemeTokenizer.load_from_file(vocab_file)
        elif mode.startswith("bert-"):
            return BertTokenizer.load_from_file(mode)
        else:
            raise NotImplementedError("`{}` is not yet supported.".format(mode))


def default_phoneme_tokenizer() -> PhonemeTokenizer:
    """Returns a default LibriSpeech phoneme tokenizer.

    Returns:
        PhonemeTokenizer: Vocabs include 71 phonemes
    """

    return PhonemeTokenizer.load_from_file(vocab_list=PHONEME_VOCAB)
