"""Downstream expert for Query-by-Example Spoken Term Detection on QUESST 2014."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dtw import dtw
from lxml import etree
from scipy.spatial import distance
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import QUESST14Dataset


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(
        self, upstream_dim: int, downstream_expert: dict, expdir: str, **kwargs
    ):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.max_workers = downstream_expert["max_workers"]
        self.feature_normalization = downstream_expert["feature_normalization"]
        self.silence_frame = downstream_expert["silence_frame"]
        self.datarc = downstream_expert["datarc"]
        self.dtwrc = downstream_expert["dtwrc"]
        self.expdir = Path(expdir)
        self.test_dataset = None

        assert not (
            self.feature_normalization and self.dtwrc["dist_method"] == "cosine_neg_log"
        ), "Upstream features normalization cannot be used with cosine_neg_log."

        assert (
            self.dtwrc["step_pattern"] == "asymmetric" or not self.dtwrc["subsequence"]
        ), "Subsequence finding only works under asymmetric setting."

    # Interface
    def get_dataloader(self, mode):
        if mode == "dev":
            self.test_dataset = QUESST14Dataset("dev", **self.datarc)
        else:  # eval
            self.test_dataset = QUESST14Dataset("eval", **self.datarc)

        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.datarc["batch_size"],
            drop_last=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=self.test_dataset.collate_fn,
        )

    # Interface
    def forward(
        self,
        mode,
        features,
        audio_names,
        records,
        **kwargs,
    ):
        for feature, audio_name in zip(features, audio_names):
            feature = feature.detach().cpu()
            if self.silence_frame is not None:  # remove silence frames
                feature = feature[feature.argmax(1) != self.silence_frame]
            records["features"].append(feature)
            records["audio_names"].append(audio_name)

    # interface
    def log_records(self, mode, records, **kwargs):
        """Perform DTW and save results."""

        # Get precomputed queries & docs
        queries = records["features"][: self.test_dataset.n_queries]
        docs = records["features"][self.test_dataset.n_queries :]
        query_names = records["audio_names"][: self.test_dataset.n_queries]
        doc_names = records["audio_names"][self.test_dataset.n_queries :]

        # Normalize upstream features
        feature_mean, feature_std = 0.0, 1.0
        if self.feature_normalization:
            feats = torch.cat(records["features"])
            feature_mean = feats.mean(0)
            feature_std = torch.clamp(feats.std(0), 1e-9)
        queries = [((query - feature_mean) / feature_std).numpy() for query in queries]
        docs = [((doc - feature_mean) / feature_std).numpy() for doc in docs]

        # Define distance function for DTW
        if self.dtwrc["dist_method"] == "cosine_exp":
            dist_fn = cosine_exp
        elif self.dtwrc["dist_method"] == "cosine_neg_log":
            dist_fn = cosine_neg_log
        else:
            dist_fn = partial(distance.cdist, metric=self.dtwrc["dist_method"])

        # Define DTW configurations
        dtwrc = {
            "step_pattern": self.dtwrc["step_pattern"],
            "keep_internals": False,
            "distance_only": False if self.dtwrc["subsequence"] else True,
            "open_begin": True if self.dtwrc["subsequence"] else False,
            "open_end": True if self.dtwrc["subsequence"] else False,
        }

        # Calculate matching scores
        results = defaultdict(list)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for query, query_name in zip(queries, query_names):
                if len(query) < 5:  # Do not consider too short queries
                    results[query_name] = [(doc_name, 0) for doc_name in doc_names]
                    continue
                for doc, doc_name in zip(docs, doc_names):
                    futures.append(
                        executor.submit(
                            match,
                            query,
                            doc,
                            query_name,
                            doc_name,
                            dist_fn,
                            self.dtwrc["minmax_norm"],
                            dtwrc,
                        )
                    )
            for future in tqdm(
                as_completed(futures), total=len(futures), ncols=0, desc="DTW"
            ):
                query_name, doc_name, score = future.result()
                results[query_name].append((doc_name, score))

        # Normalize scores with regard to each query
        for query_name, doc_scores in results.items():
            names, scores = zip(*doc_scores)
            scores = np.array(scores)
            scores = (scores - scores.mean()) / np.clip(scores.std(), 1e-9, np.inf)
            results[query_name] = list(zip(names, scores))

        # Scores above 2 STDs are seen as detected (top 2.5% as YES)
        score_thresh = 2.0

        # Build XML tree
        root = etree.Element(
            "stdlist",
            termlist_filename="benchmark.stdlist.xml",
            indexing_time="1.00",
            language="english",
            index_size="1",
            system_id="benchmark",
        )
        for query_name, doc_scores in results.items():
            term_list = etree.SubElement(
                root,
                "detected_termlist",
                termid=query_name,
                term_search_time="1.0",
                oov_term_count="1",
            )
            for doc_name, score in doc_scores:
                etree.SubElement(
                    term_list,
                    "term",
                    file=doc_name,
                    channel="1",
                    tbeg="0.000",
                    dur="0.00",
                    score=f"{score:.4f}",
                    decision="YES" if score > score_thresh else "NO",
                )

        # Output XML
        etree.ElementTree(root).write(
            str(self.expdir / "benchmark.stdlist.xml"),
            encoding="UTF-8",
            pretty_print=True,
        )


def match(query, doc, query_name, doc_name, dist_fn, minmax_norm, dtwrc):
    """Match between a query and a doc."""
    dist = dist_fn(query, doc)

    if minmax_norm:
        dist_min = dist.min(1)[:, np.newaxis]
        dist_max = dist.max(1)[:, np.newaxis]
        dist = (dist - dist_min) / np.clip(dist_max - dist_min, 1e-9, np.inf)

    dtw_result = dtw(x=dist, **dtwrc)
    cost = dtw_result.normalizedDistance
    return query_name, doc_name, -1 * cost


def cosine_exp(query, doc):
    dist = distance.cdist(query, doc, "cosine")
    dist = np.exp(dist) - 1
    return dist


def cosine_neg_log(query, doc):
    dist = distance.cdist(query, doc, "cosine")
    dist = -1 * np.log(1 - dist)
    return dist
