from s3prl.dataio.encoder.tokenizer import CharacterTokenizer, default_phoneme_tokenizer


def test_tokenizer():
    char_tokenizer = CharacterTokenizer()
    phone_tokenizer = default_phoneme_tokenizer()

    char_text = "HELLO WORLD"
    char_text_enc = char_tokenizer.encode(char_text)
    char_text_dec = char_tokenizer.decode(char_text_enc)

    assert isinstance(char_text_enc, list)
    assert char_text == char_text_dec
