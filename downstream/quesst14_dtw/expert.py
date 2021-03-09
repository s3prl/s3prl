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
        self.test_dataset = QUESST14Dataset(**self.datarc)

    # Interface
    def get_dataloader(self, mode):
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
            feature_mean, feature_std = feats.mean(0), feats.std(0)
            feature_std[feature_std == 0.0] = 1e-9
        queries = [((query - feature_mean) / feature_std).numpy() for query in queries]
        docs = [((doc - feature_mean) / feature_std).numpy() for doc in docs]

        # Define distance function for DTW
        dist_fn = (
            cosine_exp_minmax
            if self.dtwrc["dist_method"] == "cosine_exp_minmax"
            else partial(distance.cdist, metric=self.dtwrc["dist_method"])
        )

        # Define DTW configurations
        dtwrc = {
            "step_pattern": self.dtwrc["step_pattern"],
            "keep_internals": False,
            "distance_only": False if self.dtwrc["subsequence"] else True,
            "open_begin": True if self.dtwrc["subsequence"] else False,
            "open_end": True if self.dtwrc["subsequence"] else False,
        }
        assert (
            self.dtwrc["step_pattern"] == "asymmetric" or not self.dtwrc["subsequence"]
        )  # subsequence finding only works under asymmetric setting

        # Calculate matching scores
        results = defaultdict(list)
        scores = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for query, query_name in zip(queries, query_names):
                for doc, doc_name in zip(docs, doc_names):
                    futures.append(
                        executor.submit(
                            match, query, doc, query_name, doc_name, dist_fn, dtwrc
                        )
                    )
            for future in tqdm(
                as_completed(futures), total=len(futures), ncols=0, desc="DTW"
            ):
                query_name, doc_name, score = future.result()
                results[query_name].append((doc_name, score))
                scores.append(score)

        # Determine score threshold (top 1% as YES, the rest as NO)
        scores = sorted(scores)
        score_thresh = scores[int(0.99 * len(scores))]
        score_min = scores[0]

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
                    score=f"{score - score_min:.4f}",
                    decision="YES" if score > score_thresh else "NO",
                )

        # Output XML
        etree.ElementTree(root).write(
            str(self.expdir / "benchmark.stdlist.xml"),
            encoding="UTF-8",
            pretty_print=True,
        )


def match(query, doc, query_name, doc_name, dist_fn, dtwrc):
    """Match between a query and a doc."""
    dist = dist_fn(query, doc)
    dtw_result = dtw(x=dist, **dtwrc)
    cost = dtw_result.normalizedDistance
    return query_name, doc_name, -1 * cost


def cosine_exp_minmax(query, doc):
    dist = distance.cdist(query, doc, "cosine")
    dist = np.exp(dist) - 1
    dist_min, dist_max = dist.min(1), dist.max(1)
    dist = ((dist.T - dist_min) / (dist_max - dist_min)).T
    return dist
