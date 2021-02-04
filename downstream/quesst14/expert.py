"""Downstream expert for Query-by-Example Spoken Term Detection on QUESST 2014."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lxml import etree
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .dataset import QUESST14Dataset, SWS2013Dataset
from .model import Model


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
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = Path(expdir)
        self.train_dataset = SWS2013Dataset(**self.datarc)

        self.model = Model(
            input_dim=upstream_dim,
            **self.modelrc,
        )
        self.objective = nn.MSELoss()

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.datarc["batch_size"],
            drop_last=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=WeightedRandomSampler(
                self.train_dataset.sample_weights,
                len(self.train_dataset.sample_weights),
            ),
            batch_size=self.datarc["batch_size"],
            drop_last=True,
            num_workers=self.datarc["num_workers"],
            collate_fn=self.train_dataset.collate_fn,
        )

    # Interface
    def get_dev_dataloader(self):
        return None

    # Interface
    def get_test_dataloader(self):
        return None

    # Interface
    def forward(
        self,
        features,
        labels,
        records,
        **kwargs,
    ):
        audio_tensors = torch.stack(features[: len(features) // 2])
        query_tensors = torch.stack(features[len(features) // 2 :])
        labels = torch.stack(labels).to(audio_tensors.device)
        # print(audio_tensors.shape, query_tensors.shape, labels.shape)
        audio_embs = self.model(audio_tensors)
        query_embs = self.model(query_tensors)
        # print(audio_embs.shape, query_embs.shape)
        similarities = torch.sum(audio_embs * query_embs, dim=-1, keepdim=True)
        # print(similarities.shape)
        return self.objective(similarities, labels)

    # interface
    def log_records(self, records, **kwargs):
        """Perform DTW and save results."""
        pass
        # queries = records["features"][: self.test_dataset.n_queries]
        # docs = records["features"][self.test_dataset.n_queries :]
        # query_names = records["audio_names"][: self.test_dataset.n_queries]
        # doc_names = records["audio_names"][self.test_dataset.n_queries :]
        # results = defaultdict(list)
        # scores = []

        # # Calculate matching scores
        # with ProcessPoolExecutor(self.datarc["num_workers"]) as executor:
        #     futures = []

        #     for query, query_name in zip(queries, query_names):
        #         query_name = query_name.replace(".wav", "")
        #         for doc, doc_name in zip(docs, doc_names):
        #             doc_name = doc_name.replace(".wav", "")
        #             futures.append(
        #                 executor.submit(match, query, doc, query_name, doc_name)
        #             )

        #     for future in tqdm(
        #         as_completed(futures), total=len(futures), ncols=0, desc="DTW"
        #     ):
        #         query_name, doc_name, score = future.result()
        #         results[query_name].append((doc_name, score))
        #         scores.append(score)

        # # Determine score threshold
        # scores = sorted(scores)
        # score_thresh = scores[int(0.99 * len(scores))]
        # score_min = scores[0]

        # # Build XML tree
        # root = etree.Element(
        #     "stdlist",
        #     termlist_filename="benchmark.stdlist.xml",
        #     indexing_time="1.00",
        #     language="english",
        #     index_size="1",
        #     system_id="benchmark",
        # )
        # for query_name, doc_scores in results.items():
        #     term_list = etree.SubElement(
        #         root,
        #         "detected_termlist",
        #         termid=query_name,
        #         term_search_time="1.0",
        #         oov_term_count="1",
        #     )
        #     for doc_name, score in doc_scores:
        #         etree.SubElement(
        #             term_list,
        #             "term",
        #             file=doc_name,
        #             channel="1",
        #             tbeg="0.000",
        #             dur="0.00",
        #             score=f"{score - score_min:.4f}",
        #             decision="YES" if score > score_thresh else "NO",
        #         )

        # # Output XML
        # tree = etree.ElementTree(root)
        # tree.write(
        #     str(self.expdir / "benchmark.stdlist.xml"),
        #     encoding="UTF-8",
        #     pretty_print=True,
        # )


# def match(query, doc, query_name, doc_name):
#     cost = segmental_dtw(query, doc)
#     return query_name, doc_name, -1 * cost
