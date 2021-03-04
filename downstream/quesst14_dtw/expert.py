"""Downstream expert for Query-by-Example Spoken Term Detection on QUESST 2014."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn as nn
from lxml import etree
from torch.utils.data import DataLoader
from tqdm import tqdm
from dtw import dtw

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
        self.max_workers = (
            downstream_expert["max_workers"]
            if downstream_expert["max_workers"] > 0
            else None
        )
        self.datarc = downstream_expert["datarc"]
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
            records["features"].append(feature)
            records["audio_names"].append(audio_name)

    # interface
    def log_records(self, mode, records, **kwargs):
        """Perform DTW and save results."""
        queries = records["features"][: self.test_dataset.n_queries]
        docs = records["features"][self.test_dataset.n_queries :]
        query_names = records["audio_names"][: self.test_dataset.n_queries]
        doc_names = records["audio_names"][self.test_dataset.n_queries :]

        # Normalize representations
        feats = torch.cat(records["features"])
        feature_mean = feats.mean(0)
        feature_std = feats.std(0)
        feature_std[feature_std == 0.0] = 1e-9
        queries = [((query - feature_mean) / feature_std).numpy() for query in queries]
        docs = [((doc - feature_mean) / feature_std).numpy() for doc in docs]

        # Store results
        results = defaultdict(list)
        scores = []

        # Calculate matching scores
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for query, query_name in zip(queries, query_names):
                for doc, doc_name in zip(docs, doc_names):
                    futures.append(
                        executor.submit(
                            match,
                            query,
                            doc,
                            query_name,
                            doc_name,
                        )
                    )

            for future in tqdm(
                as_completed(futures), total=len(futures), ncols=0, desc="DTW"
            ):
                query_name, doc_name, score = future.result()
                results[query_name].append((doc_name, score))
                scores.append(score)

        # Determine score threshold
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
        tree = etree.ElementTree(root)
        tree.write(
            str(self.expdir / "benchmark.stdlist.xml"),
            encoding="UTF-8",
            pretty_print=True,
        )


def match(query, doc, query_name, doc_name):
    dtw_result = dtw(
        x=query,
        y=doc,
        # dist_method="correlation",
        dist_method="cosine",
        # dist_method="cityblock",
        # dist_method="euclidean",
        # dist_method="sqeuclidean",
        # step_pattern="symmetric2",
        step_pattern="asymmetric",
        # step_pattern="rigid",
        keep_internals=False,
        # distance_only=True,
        distance_only=False,
        open_end=True,
        # open_begin=False,
        open_begin=True,
    )
    cost = dtw_result.normalizedDistance
    return query_name, doc_name, -1 * cost
