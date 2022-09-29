import logging

from s3prl.dataio.encoder.g2p import G2P


def test_g2p():
    g2p = G2P()
    char_sent = "HELLO WORLD"
    phn_sent = g2p.encode(char_sent)
    logging.info(phn_sent)
