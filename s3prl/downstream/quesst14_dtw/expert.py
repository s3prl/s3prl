"""Downstream expert for Query-by-Example Spoken Term Detection on QUESST 2014."""

import pickle
import logging
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
from .dtw_utils import dtw_and_dump_tree

log = logging.getLogger(__name__)

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
        self.two_stages = downstream_expert["two_stages"]
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

        if self.two_stages:
            log.info("Saving features for later DTW...")
            def dump_variables(name: str, variable):
                with open(self.expdir / f"{name}.pkl", "wb") as file:
                    pickle.dump(variable, file)

            dump_variables("queries", queries)
            dump_variables("docs", docs)
            dump_variables("query_names", query_names)
            dump_variables("doc_names", doc_names)
        else:
            log.info("Running DTW...")
            dtw_and_dump_tree(
                queries,
                query_names,
                docs,
                doc_names,
                self.dtwrc,
                self.expdir,
                self.max_workers,
                self.feature_normalization
            )
