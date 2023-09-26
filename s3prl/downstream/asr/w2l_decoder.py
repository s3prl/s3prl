#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Flashlight decoders.
"""

import itertools as it
from typing import List
import warnings

import torch

try:
    from flashlight.lib.text.dictionary import create_word_dict, load_words
    from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        SmearingMode,
        Trie,
        LexiconDecoder,
    )
except:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. Please install from https://github.com/flashlight/text and https://github.com/flashlight/sequence"
    )
    LM = object
    LMState = object


class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        if "<sep>" in tgt_dict.indices:
            self.silence = tgt_dict.index("<sep>")
        elif "|" in tgt_dict.indices:
            self.silence = tgt_dict.index("|")
        else:
            self.silence = tgt_dict.eos()
        self.asg_transitions = None

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        model = models[0]
        encoder_out = model(**encoder_input)
        if hasattr(model, "get_logits"):
            emissions = model.get_logits(encoder_out)  # no need to normalize emissions
        else:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))


class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.unit_lm = getattr(args, "unit_lm", False)

        if args.lexicon:
            self.lexicon = load_words(args.lexicon)
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")

            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]
                    assert (
                        tgt_dict.unk() not in spelling_idxs
                    ), f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                word_score=args.word_score,
                unk_score=args.unk_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )

            if self.asg_transitions is None:
                N = 768
                # self.asg_transitions = torch.FloatTensor(N, N).zero_()
                self.asg_transitions = []

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                self.asg_transitions,
                self.unit_lm,
            )
        else:
            assert (
                args.unit_lm
            ), "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import (
                LexiconFreeDecoder,
                LexiconFreeDecoderOptions,
            )

            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

    def get_timesteps(self, token_idxs: List[int]) -> List[int]:
        """Returns frame numbers corresponding to every non-blank token.

        Parameters
        ----------
        token_idxs : List[int]
            IDs of decoded tokens.

        Returns
        -------
        List[int]
            Frame numbers corresponding to every non-blank token.
        """
        timesteps = []
        for i, token_idx in enumerate(token_idxs):
            if token_idx == self.blank:
                continue
            if i == 0 or token_idx != token_idxs[i - 1]:
                timesteps.append(i)
        return timesteps

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(result.tokens),
                        "score": result.score,
                        "timesteps": self.get_timesteps(result.tokens),
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    }
                    for result in nbest_results
                ]
            )
        return hypos
