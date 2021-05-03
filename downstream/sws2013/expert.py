"""Downstream expert for Query-by-Example Spoken Term Detection on SWS 2013."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from lxml import etree
from tqdm import tqdm

from .sws2013_dataset import SWS2013Dataset
from .sws2013_testset import SWS2013Testset
from .quesst14_dataset import QUESST14Dataset
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
        self.lossrc = downstream_expert["lossrc"]

        # Result dir setup, used to save output XML file
        self.expdir = Path(expdir)

        # Dataset, model, loss setup
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = Model(
            input_dim=upstream_dim,
            **self.modelrc,
        )
        self.objective = nn.CosineEmbeddingLoss(**self.lossrc)

    # Interface
    def get_dataloader(self, mode):
        if mode == "train":
            self.train_dataset = SWS2013Dataset("dev", **self.datarc)
            self.valid_dataset = SWS2013Dataset("eval", **self.datarc)

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
            self.test_dataset = QUESST14Dataset(mode, **self.datarc)
            return DataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.datarc["batch_size"],
                drop_last=False,
                num_workers=self.datarc["num_workers"],
                collate_fn=self.test_dataset.collate_fn,
            )

        if mode == "sws2013_eval":
            self.test_dataset = SWS2013Testset("eval", **self.datarc)
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
        labels,
        records,
        **kwargs,
    ):
        if mode in ["train", "valid"]:
            audio_tensors = torch.stack(features[: len(features) // 2])
            query_tensors = torch.stack(features[len(features) // 2 :])
            labels = torch.cat(labels).to(audio_tensors.device)

            audio_embs = self.model(audio_tensors)
            query_embs = self.model(query_tensors)

            # cosine embedding loss
            loss = self.objective(audio_embs, query_embs, labels)
            records["loss"].append(loss.item())

            with torch.no_grad():
                # cosine similarity
                similarities = F.cosine_similarity(audio_embs, query_embs)
                records["similarity-positive"] += similarities[labels > 0].tolist()
                records["similarity-negative"] += similarities[labels < 0].tolist()

            return loss

        elif mode in ["dev", "eval", "sws2013_eval"]:
            audio_tensors = torch.stack(features)
            lengths, audio_names = labels

            embs = self.model(audio_tensors)
            embs = embs.detach().cpu()

            offset = 0
            for length, audio_name in zip(lengths, audio_names):
                records["embs"].append(embs[offset : offset + length])
                records["audio_names"].append(audio_name)
                offset += length

        else:
            raise NotImplementedError

    # Interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        """Log training, validation information or test on a dataset."""

        if mode in ["train", "valid"]:
            prefix = f"sws2013/{mode}"
            for key, val in records.items():
                average = sum(val) / len(val)
                logger.add_scalar(f"{prefix}-{key}", average, global_step=global_step)

        elif mode in ["dev", "eval", "sws2013_eval"]:
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
                scores = (scores - scores.mean()) / (scores.std() + 1e-6)
                results[query_name] = list(zip(doc_names, scores.tolist()))

            # Determine score threshold
            score_thresh = 0

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
            tree = etree.ElementTree(root)
            tree.write(
                str(self.expdir / "benchmark.stdlist.xml"),
                encoding="UTF-8",
                pretty_print=True,
            )

        else:
            raise NotImplementedError
