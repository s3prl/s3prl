from s3prl.util.tokenizer import load_tokenizer
from s3prl.preprocessor.librispeech_ctc_preprocessor import VOCAB_LIST


def test_tokenizer():
    char_tokenizer = load_tokenizer("character", vocab_list=VOCAB_LIST["character"])
    phone_tokenizer = load_tokenizer("phoneme", vocab_list=VOCAB_LIST["phoneme"])

    char_text = "HELLO WORLD"
    char_text_enc = char_tokenizer.encode(char_text)
    char_text_dec = char_tokenizer.decode(char_text_enc)

    assert isinstance(char_text_enc, list)
    assert char_text == char_text_dec
