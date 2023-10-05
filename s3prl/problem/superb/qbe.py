import logging
import os
import re
import subprocess
import tempfile
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
from tqdm import tqdm

from s3prl import Container
from s3prl.corpus.quesst14 import quesst14_for_qbe
from s3prl.dataset.base import AugmentedDynamicItemDataset, DataPipe, SequentialDataPipe
from s3prl.dataset.common_pipes import LoadAudio, SetOutputKeys
from s3prl.sampler import FixedBatchSizeBatchSampler
from s3prl.task.dump_feature import DumpFeature
from s3prl.util import workspace
from s3prl.util.configuration import default_cfg, field
from s3prl.util.workspace import Workspace, as_type

from .base import SuperbProblem

logger = logging.getLogger(__name__)


def cosine_exp(query, doc):
    dist = distance.cdist(query, doc, "cosine")
    dist = np.exp(dist) - 1
    return dist


def cosine_neg_log(query, doc):
    dist = distance.cdist(query, doc, "cosine")
    dist = -1 * np.log(1 - dist)
    return dist


class QbeDumpFeaturePipe(DataPipe):
    def __init__(
        self,
        output_keys: dict = None,
        sox_effects: list = None,
    ):
        output_keys = output_keys or dict(
            x="wav",
            x_len="wav_len",
            unique_name="id",
        )
        self.pipes = SequentialDataPipe(
            LoadAudio(sox_effects=sox_effects),
            SetOutputKeys(output_keys=output_keys),
        )

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        return self.pipes(dataset)


class SuperbQBE(SuperbProblem):
    @default_cfg(
        workspace="???",
        corpus=dict(
            CLS=quesst14_for_qbe,
            dataset_root="???",
        ),
        all_datapipe=dict(
            CLS=QbeDumpFeaturePipe,
            sox_effects=[
                ["channels", "1"],
                ["rate", "16000"],
                ["gain", "-3.0"],
            ],
        ),
        all_sampler=dict(
            CLS=FixedBatchSizeBatchSampler,
            batch_size=1,
        ),
        upstream=dict(
            CLS="S3PRLUpstream",
            name="???",
        ),
        task=dict(
            CLS=DumpFeature,
        ),
    )
    @classmethod
    def setup(cls, **cfg):
        cfg = Container(cfg)
        workspace = Workspace(cfg.workspace)

        if not isinstance(cfg.upstream, nn.Module):
            model = cfg.upstream.CLS(**cfg.upstream.kwds())
        else:
            model = cfg.upstream

        logger.info("Preparing corpus")
        all_data, valid_query_keys, test_query_keys, doc_keys = cfg.corpus.CLS(
            **cfg.corpus.kwds()
        ).slice(4)

        logger.info("Preparing train data")
        all_dataset = AugmentedDynamicItemDataset(all_data)
        all_dataset = SequentialDataPipe(*cfg.all_datapipe.tolist())(all_data)
        all_sampler = cfg.all_sampler.CLS(all_dataset, **cfg.all_sampler.kwds())

        task = cfg.task.CLS(model, **cfg.task.kwds())

        workspace.update(
            dict(
                all_dataset=all_dataset,
                all_sampler=all_sampler,
                task=task,
                valid_query_keys=as_type(valid_query_keys, "yaml"),
                test_query_keys=as_type(test_query_keys, "yaml"),
                doc_keys=as_type(doc_keys, "yaml"),
            )
        )

        # This is for easy reuse the inference command for feature extraction
        workspace.link_from("valid_best_task", workspace, "task")

    @default_cfg(
        **SuperbProblem.inference.default_except(
            split_name="all",
        )
    )
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @default_cfg(
        workspace=field(
            "???",
            "Should have 'feat' sub-workspace, 'valid_query_keys', 'test_query_keys', and 'doc_keys'",
        ),
        dtw=dict(),
        doc_num=field(
            -1,
            "Only take the first 'doc_num' docs to be searched by the queries. Set -1 to disable",
        ),
    )
    @classmethod
    def dtw_for_quesst14(cls, **cfg):
        cfg = Container(cfg)
        workspace = Workspace(cfg.workspace)

        feat_dir = workspace / "feat"
        assert len(feat_dir.files()) > 0, f"No files in {feat_dir}"
        num_layers = feat_dir[feat_dir.files()[0]].shape[0]

        valid_query_keys = workspace["valid_query_keys"]
        doc_keys = workspace["doc_keys"]
        if cfg.doc_num != -1:
            doc_keys = doc_keys[: cfg.doc_num]

        layer_mtwv = {}
        scoring_dir = (
            Workspace(workspace.get_cfg(cls.setup).corpus.dataset_root) / "scoring"
        )

        layers = cfg.layers or range(num_layers)
        for layer_id in layers:
            queries = []
            for key in tqdm(
                valid_query_keys, desc=f"Load valid query features for layer {layer_id}"
            ):
                queries.append(torch.from_numpy(feat_dir[key][layer_id]))

            docs = []
            for key in tqdm(doc_keys, desc=f"Load doc features for layer {layer_id}"):
                docs.append(torch.from_numpy(feat_dir[key][layer_id]))

            valid_results = cls.dtw(
                queries, valid_query_keys, docs, doc_keys, **cfg.dtw.kwds()
            )

            layer_dir = workspace / f"valid_layer_{layer_id}"
            metrics = cls._scoring(valid_results, layer_dir, scoring_dir, is_valid=True)
            layer_dir.put(metrics, "metrics", "yaml")
            layer_mtwv[layer_id] = metrics.maxTWV
        del queries
        del docs

        layer_mtwv = [(layer_id, mtwv) for layer_id, mtwv in layer_mtwv.items()]
        layer_mtwv.sort(key=lambda x: x[1], reverse=True)
        logger.info("Sorted all-layer results:")
        for layer_id, mtwv in layer_mtwv:
            logger.info(f"Layer {layer_id} valid maxTWV: {mtwv}")

        best_layer_id = layer_mtwv[0][0]
        logger.info(f"The best valid layer: {best_layer_id}")

        test_query_keys = workspace["test_query_keys"]
        queries = []
        for key in tqdm(
            test_query_keys, desc=f"Load test query features for layer {best_layer_id}"
        ):
            queries.append(torch.from_numpy(feat_dir[key][best_layer_id]))

        docs = []
        for key in tqdm(doc_keys, desc=f"Load doc features for layer {best_layer_id}"):
            docs.append(torch.from_numpy(feat_dir[key][best_layer_id]))

        test_results = cls.dtw(
            queries, test_query_keys, docs, doc_keys, **cfg.dtw.kwds()
        )
        layer_dir = workspace / f"test_layer_{best_layer_id}"
        metrics = cls._scoring(test_results, layer_dir, scoring_dir, is_valid=False)
        layer_dir.put(metrics, "test_metrics", "yaml")
        workspace.link_from("valid_best_metrics", layer_dir, "test_metrics")
        logger.info(
            f"The best valid layer's (layer {best_layer_id}) test maxTWV: {metrics.maxTWV}"
        )

    @default_cfg(
        **SuperbProblem.run.default_except(
            stages=["setup", "inference", "dtw_for_quesst14"],
            start_stage="setup",
            final_stage="dtw_for_quesst14",
            setup=setup.default_cfg.deselect("workspace", "resume", "dryrun"),
            inference=inference.default_cfg.deselect("workspace", "dryrun"),
            dtw_for_quesst14=dtw_for_quesst14.default_cfg.deselect("workspace"),
        )
    )
    @classmethod
    def run(cls, **cfg):
        super().run(**cfg)

    @classmethod
    def dtw(
        cls,
        queries,
        queries_name,
        docs,
        doc_names,
        feature_normalization: bool = True,
        dist_method: str = "cosine_exp",
        step_pattern: str = "asymmetric",
        minmax_norm: bool = True,
        subsequence: bool = True,
        n_jobs: int = 12,
    ):
        """
        Return:
            results (dict):
                key is query name, value is a list of (doc_name, doc_score) where score is higher better
        """
        # Normalize upstream features
        feature_mean, feature_std = 0.0, 1.0
        if feature_normalization:
            feats = torch.cat([*queries, *docs])
            feature_mean = feats.mean(0)
            feature_std = torch.clamp(feats.std(0), 1e-9)
        queries = [((query - feature_mean) / feature_std).numpy() for query in queries]
        docs = [((doc - feature_mean) / feature_std).numpy() for doc in docs]

        # Define distance function for DTW
        if dist_method == "cosine_exp":
            dist_fn = cosine_exp
        elif dist_method == "cosine_neg_log":
            dist_fn = cosine_neg_log
        else:
            dist_fn = partial(distance.cdist, metric=dist_method)

        # Define DTW configurations
        dtwrc = {
            "step_pattern": step_pattern,
            "keep_internals": False,
            "distance_only": False if subsequence else True,
            "open_begin": True if subsequence else False,
            "open_end": True if subsequence else False,
        }

        # Calculate matching scores
        results = defaultdict(list)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for query, query_name in zip(queries, queries_name):
                if len(query) < 5:  # Do not consider too short queries
                    results[query_name] = [(doc_name, 0) for doc_name in doc_names]
                    continue
                for doc, doc_name in zip(docs, doc_names):
                    futures.append(
                        executor.submit(
                            cls.match,
                            query,
                            doc,
                            query_name,
                            doc_name,
                            dist_fn,
                            minmax_norm,
                            dtwrc,
                        )
                    )
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                dynamic_ncols=True,
                desc="dtw",
            ):
                query_name, doc_name, score = future.result()
                results[query_name].append((doc_name, score))

        # Normalize scores with regard to each query
        for query_name, doc_scores in results.items():
            names, scores = zip(*doc_scores)
            scores = np.array(scores)
            scores = (scores - scores.mean()) / np.clip(scores.std(), 1e-9, np.inf)
            results[query_name] = list(zip(names, scores))

        return results

    @classmethod
    def match(cls, query, doc, query_name, doc_name, dist_fn, minmax_norm, dtwrc):
        """Match between a query and a doc."""
        dist = dist_fn(query, doc)

        if minmax_norm:
            dist_min = dist.min(1)[:, np.newaxis]
            dist_max = dist.max(1)[:, np.newaxis]
            dist = (dist - dist_min) / np.clip(dist_max - dist_min, 1e-9, np.inf)

        dtw_result = dtw(x=dist, **dtwrc)
        cost = dtw_result.normalizedDistance
        return query_name, doc_name, -1 * cost

    @classmethod
    def _scoring(
        cls,
        results,
        workspace: Workspace,
        scoring_dir: Workspace,
        is_valid: bool = True,
    ):
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

        workspace.mkdir(exist_ok=True, parents=True)
        xml_path = str((workspace / "benchmark.stdlist.xml").resolve())
        etree.ElementTree(root).write(
            xml_path,
            encoding="UTF-8",
            pretty_print=True,
        )

        current_dir = os.getcwd()
        os.chdir(str(scoring_dir))

        target = "groundtruth_quesst14_dev" if is_valid else "groundtruth_quesst14_eval"
        try:
            result = subprocess.check_output(
                f"./score-TWV-Cnxe.sh {Path(xml_path).parent} {target} -10",
                shell=True,
            ).decode("utf-8")
        except subprocess.CalledProcessError as e:
            result = e.output.decode("utf-8")

        assert "maxTWV" in result
        actTWV, maxTWV, threshold = re.search(
            "actTWV: (.+) maxTWV: (.+) Threshold: (.+)\n",
            result,
        ).groups()

        os.chdir(current_dir)
        return Container(
            actTWV=float(actTWV.strip()),
            maxTWV=float(maxTWV.strip()),
            threshold=float(threshold.strip()),
        )
