import os
from collections import Counter
from multiprocessing import get_context

import torch

from fairseq.data.dictionary import Dictionary as fairseq_Dictionary


class Dictionary(fairseq_Dictionary):
    """Dictionary inheritted from FairSeq"""

    @staticmethod
    def _add_transcripts_to_dictionary_single_worker(
        transcripts, eos_word, worker_id=0, num_workers=1
    ):
        counter = Counter()
        size = len(transcripts)
        chunk_size = size // num_workers
        offset = worker_id * chunk_size
        end = min(size + 1, offset + chunk_size)
        for line in transcripts[offset:end]:
            for word in line.split():
                counter.update([word])
            counter.update([eos_word])
        return counter

    @staticmethod
    def add_transcripts_to_dictionary(transcripts, dict, num_workers):
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = get_context('spawn').Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_transcripts_to_dictionary_single_worker,
                        (transcripts, dict.eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_transcripts_to_dictionary_single_worker(
                    transcripts, dict.eos_word
                )
            )