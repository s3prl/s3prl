"""Downstream expert for Query-by-Example Spoken Term Detection on SWS 2013."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from lxml import etree
from tqdm import tqdm

from .quesst14_trainset import QUESST14Trainset
from .quesst14_testset import QUESST14Testset
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

        # Config setup
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        # Result dir setup, used to save output XML file
        self.expdir = Path(expdir)

        # Dataset, model, loss setup
        self.train_dataset = QUESST14Trainset("dev", **self.datarc)
        self.valid_dataset = QUESST14Trainset("eval", **self.datarc)
        self.test_dataset = None
        self.model = Model(
            input_dim=upstream_dim,
            **self.modelrc,
        )

    # Interface
    def get_dataloader(self, mode):
        if mode == "train":
            return DataLoader(
                self.train_dataset,
                sampler=WeightedRandomSampler(
                    weights=self.train_dataset.sample_weights,
                    num_samples=len(self.train_dataset.sample_weights),
                    replacement=True,
                ),
                batch_size=self.datarc["batch_size"],
                drop_last=True,
                num_workers=self.datarc["num_workers"],
                collate_fn=self.train_dataset.collate_fn,
            )

        if mode == "valid":
            return DataLoader(
                self.valid_dataset,
                sampler=WeightedRandomSampler(
                    weights=self.valid_dataset.sample_weights,
                    num_samples=self.datarc["valid_size"],
                    replacement=True,
                ),
                batch_size=self.datarc["batch_size"],
                drop_last=True,
                num_workers=self.datarc["num_workers"],
                collate_fn=self.valid_dataset.collate_fn,
            )

        if mode in ["dev", "eval"]:
            self.test_dataset = QUESST14Testset(mode, **self.datarc)
            return DataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.datarc["batch_size"],
                drop_last=False,
                num_workers=self.datarc["num_workers"],
                collate_fn=self.test_dataset.collate_fn,
            )

        raise NotImplementedError

    # Interface
    def forward(
        self,
        mode,
        features,
        infos,
        records,
        **kwargs,
    ):
        if mode in ["train", "valid"]:
            features = torch.stack(features)
            prefix_sums, labels = infos
            labels = torch.cat(labels).to(features.device)

            embs = self.model(features)
            query_embs = embs[: self.datarc["batch_size"]]
            audio_embs = embs[self.datarc["batch_size"] :]
            max_similarities = torch.empty(len(labels)).to(labels.device)
            for i in range(self.datarc["batch_size"]):
                similarities = F.cosine_similarity(
                    query_embs[i : i + 1],
                    audio_embs[prefix_sums[i] : prefix_sums[i + 1]],
                )
                max_similarities[i] = similarities.max()

            # cosine embedding loss
            pos_similarities = max_similarities[labels > 0]
            neg_similarities = max_similarities[labels < 0]
            pos_loss = (1 - pos_similarities).sum()
            neg_loss = neg_similarities.clamp(0).sum()
            loss = (pos_loss + neg_loss) / self.datarc["batch_size"]

            # record information
            records["loss"].append(loss.item())
            records["similarity-positive"] += pos_similarities.tolist()
            records["similarity-negative"] += neg_similarities.tolist()

            return loss

        elif mode in ["dev", "eval"]:
            audio_tensors = torch.stack(features)
            prefix_sums, audio_names = infos

            embs = self.model(audio_tensors)
            embs = embs.detach().cpu()

            for i in range(len(audio_names)):
                records["embs"].append(embs[prefix_sums[i] : prefix_sums[i + 1]])
                records["audio_names"].append(audio_names[i])

        else:
            raise NotImplementedError

    # Interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        """Log training, validation information or test on a dataset."""

        if mode in ["train", "valid"]:
            prefix = f"quesst14_embedding/{mode}"
            for key, val in records.items():
                average = sum(val) / len(val)
                logger.add_scalar(f"{prefix}-{key}", average, global_step=global_step)

        elif mode in ["dev", "eval"]:
            query_embs = records["embs"][: self.test_dataset.n_queries]
            doc_embs = records["embs"][self.test_dataset.n_queries :]
            query_names = records["audio_names"][: self.test_dataset.n_queries]
            doc_names = records["audio_names"][self.test_dataset.n_queries :]

            results = {}

            # Calculate matching scores
            for query_emb, query_name in zip(
                tqdm(query_embs, desc="Query", ncols=0), query_names
            ):
                query_emb = query_emb[0:1].cuda()
                scores = []

                for doc_emb, doc_name in zip(
                    tqdm(doc_embs, desc="Doc", ncols=0, leave=False), doc_names
                ):
                    with torch.no_grad():
                        doc_emb = doc_emb.cuda()
                        similarities = F.cosine_similarity(query_emb, doc_emb)
                        score = similarities.max().detach().cpu()

                    scores.append(score)

                scores = torch.stack(scores)
                if scores.std() < 0.1:
                    scores = torch.zeros_like(scores)
                else:
                    scores = (scores - scores.mean()) / (scores.std() + 1e-6) + 0.5
                results[query_name] = list(zip(doc_names, scores.tolist()))

            # Determine score threshold
            score_thresh = 0.5

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

        else:
            raise NotImplementedError
