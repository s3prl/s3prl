"""
The beam search decoder of flashlight

Authors:
  * Heng-Jui Chang 2022
"""

import itertools as it
import logging
import math
from typing import Iterable, List

import torch

from s3prl.util.download import _urls_to_filepaths

TOKEN_URL = "https://huggingface.co/datasets/s3prl/flashlight/raw/main/lexicon/librispeech_char_tokens.txt"

LEXICON_URL_1 = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst"
LEXICON_URL_2 = "https://huggingface.co/datasets/s3prl/flashlight/raw/main/lexicon/librispeech_lexicon.lst"

LM_URL_1 = "https://www.openslr.org/resources/11/4-gram.arpa.gz"
LM_URL_2 = (
    "https://huggingface.co/datasets/s3prl/flashlight/resolve/main/lm/4-gram.arpa.gz"
)

logger = logging.getLogger(__name__)

__all__ = ["BeamDecoder"]


class BeamDecoder(object):
    """Beam decoder powered by flashlight.

    Args:
        token (str, optional): Path to dictionary file. Defaults to "".
        lexicon (str, optional): Path to lexicon file. Defaults to "".
        lm (str, optional): Path to KenLM file. Defaults to "".
        nbest (int, optional): Returns nbest hypotheses. Defaults to 1.
        beam (int, optional): Beam size. Defaults to 5.
        beam_size_token (int, optional): Token beam size. Defaults to -1.
        beam_threshold (float, optional): Beam search log prob threshold. Defaults to 25.0.
        lm_weight (float, optional): language model weight. Defaults to 2.0.
        word_score (float, optional): score for words appearance in the transcription. Defaults to -1.0.
        unk_score (float, optional): score for unknown word appearance in the transcription. Defaults to -math.inf.
        sil_score (float, optional): score for silence appearance in the transcription. Defaults to 0.0.
    """

    def __init__(
        self,
        token: str = "",
        lexicon: str = "",
        lm: str = "",
        nbest: int = 1,
        beam: int = 5,
        beam_size_token: int = -1,
        beam_threshold: float = 25.0,
        lm_weight: float = 2.0,
        word_score: float = -1.0,
        unk_score: float = -math.inf,
        sil_score: float = 0.0,
    ):
        try:
            from flashlight.lib.text.decoder import (
                CriterionType,
                KenLM,
                LexiconDecoder,
                LexiconDecoderOptions,
                SmearingMode,
                Trie,
            )
            from flashlight.lib.text.dictionary import (
                Dictionary,
                create_word_dict,
                load_words,
            )

        except ImportError:
            logger.error(f"Please install flashlight to enable {__class__.__name__}")
            raise

        if token == "":
            token = _urls_to_filepaths(TOKEN_URL)
        if lexicon == "":
            # Try LEXICON_URL_2 if LEXICON_URL_1 did not work.
            lexicon = _urls_to_filepaths(LEXICON_URL_1)
        if lm == "":
            # Try LM_URL_2 if LM_URL_1 did not work.
            lm = _urls_to_filepaths(LM_URL_1)

        self.nbest = nbest

        self.token_dict = Dictionary(token)
        self.lexicon = load_words(lexicon)
        self.word_dict = create_word_dict(self.lexicon)

        self.lm = KenLM(lm, self.word_dict)

        self.sil_idx = self.token_dict.get_index("|")
        self.unk_idx = self.word_dict.get_index("<unk>")

        self.trie = Trie(self.token_dict.index_size(), self.sil_idx)
        start_state = self.lm.start(False)

        for word, spellings in self.lexicon.items():
            usr_idx = self.word_dict.get_index(word)
            _, score = self.lm.score(start_state, usr_idx)
            for spelling in spellings:
                spelling_idxs = [self.token_dict.get_index(tok) for tok in spelling]
                self.trie.insert(spelling_idxs, usr_idx, score)
        self.trie.smear(SmearingMode.MAX)

        if beam_size_token == -1:
            beam_size_token = self.token_dict.index_size()

        self.options = LexiconDecoderOptions(
            beam_size=beam,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            lm_weight=lm_weight,
            word_score=word_score,
            unk_score=unk_score,
            sil_score=sil_score,
            log_add=False,
            criterion_type=CriterionType.CTC,
        )

        self.blank_idx = self.token_dict.get_index("#")
        self.decoder = LexiconDecoder(
            self.options,
            self.trie,
            self.lm,
            self.sil_idx,
            self.blank_idx,
            self.unk_idx,
            [],
            False,
        )

    def get_tokens(self, idxs: Iterable) -> torch.LongTensor:
        """Normalize tokens by handling CTC blank, ASG replabels, etc.

        Args:
            idxs (Iterable): Token ID list output by self.decoder

        Returns:
            torch.LongTensor: Token ID list after normalization.
        """

        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank_idx, idxs)
        return torch.LongTensor(list(idxs))

    def get_timesteps(self, token_idxs: List[int]) -> List[int]:
        """Returns frame numbers corresponding to every non-blank token.

        Args:
            token_idxs (List[int]): IDs of decoded tokens.

        Returns:
            List[int]: Frame numbers corresponding to every non-blank token.
        """

        timesteps = []
        for i, token_idx in enumerate(token_idxs):
            if token_idx == self.blank_idx:
                continue
            if i == 0 or token_idx != token_idxs[i - 1]:
                timesteps.append(i)
        return timesteps

    def decode(self, emissions: torch.Tensor) -> List[List[dict]]:
        """Decode sequence.

        Args:
            emissions (torch.Tensor): Emission probabilities (in log scale).

        Returns:
            List[List[dict]]: Decoded hypotheses.
        """

        emissions = emissions.float().contiguous().cpu()
        B, T, N = emissions.size()

        hyps = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)
            nbest_results = results[: self.nbest]
            hyps.append(
                [
                    dict(
                        tokens=self.get_tokens(result.tokens),
                        score=result.score,
                        timesteps=self.get_timesteps(result.tokens),
                        words=[
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    )
                    for result in nbest_results
                ]
            )
        return hyps
