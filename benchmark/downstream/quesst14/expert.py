"""Downstream expert for Query-by-Example Spoken Term Detection on QUESST 2014."""

from concurrent.futures import ProcessPoolExecutor

import pyximport
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lxml import etree

pyximport.install(setup_args={"include_dirs": np.get_include()})

from .dataset import QUESST14Dataset
from .segmental_dtw import segmental_dtw


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim: int, downstream_expert: dict, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.test_dataset = QUESST14Dataset(**self.datarc)

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
        return None

    # Interface
    def get_dev_dataloader(self):
        return None

    # Interface
    def get_test_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    # Interface
    def forward(
        self, features, audio_names, records, **kwargs,
    ):
        for feature, audio_name in zip(features, audio_names):
            feature = feature.detach().cpu().numpy().astype(np.double)
            records["features"].append(feature)
            records["audio_names"].append(audio_name)

        return torch.zeros(1)

    # interface
    def log_records(self, records, **kwargs):
        """Perform DTW and save results."""
        queries = records["features"][: self.test_dataset.n_queries]
        docs = records["features"][self.test_dataset.n_queries :]
        query_names = records["audio_names"][: self.test_dataset.n_queries]
        doc_names = records["audio_names"][self.test_dataset.n_queries :]

        xml_tree = etree.Element(
            "stdlist",
            termlist_filename="benchmark.stdlist.xml",
            indexing_time="1.00",
            language="english",
            index_size="1",
            system_id="benchmark",
        )

        for query, query_name in tqdm(
            zip(queries, query_names), total=len(queries), ncols=0, desc="Queries"
        ):
            term_list = etree.SubElement(
                xml_tree,
                "detected_termlist",
                termid=query_name.replace(".wav", ""),
                term_search_time="1.0",
                oov_term_count="1",
            )

            with ProcessPoolExecutor(self.datarc["num_workers"]) as executor:
                futures = []

                for doc in docs:
                    futures.append(executor.submit(segmental_dtw, query, doc))

                for future, doc_name in zip(
                    tqdm(futures, ncols=0, position=1, leave=False, desc="Documents",),
                    doc_names,
                ):
                    cost = future.result()
                    score = -1 * cost
                    term = etree.SubElement(
                        term_list,
                        "term",
                        file=doc_name.replace(".wav", ""),
                        channel="1",
                        tbeg="0.000",
                        dur="0.00",
                        score=f"{score:.2f}",
                        decision="YES" if score > 10.0 else "NO",
                    )

        print(etree.tostring(xml_tree, encoding="UTF-8", pretty_print=True).decode())
