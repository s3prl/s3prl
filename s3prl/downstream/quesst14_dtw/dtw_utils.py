import os
import torch
import yaml
import glob
import torch
import pickle
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from dtw import dtw
from tqdm import tqdm
from lxml import etree
from scipy.spatial import distance

log = logging.getLogger(__name__)

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


def dtw_and_dump_tree(queries, query_names, docs, doc_names, dtwrc, expdir, max_workers=None, feature_normalization=True):
    expdir = Path(expdir)

    # Normalize upstream features
    feature_mean, feature_std = 0.0, 1.0
    if feature_normalization:
        feats = torch.cat([*queries, *docs])
        feature_mean = feats.mean(0)
        feature_std = torch.clamp(feats.std(0), 1e-9)
    queries = [((query - feature_mean) / feature_std).numpy() for query in queries]
    docs = [((doc - feature_mean) / feature_std).numpy() for doc in docs]

    # Define distance function for DTW
    if dtwrc["dist_method"] == "cosine_exp":
        dist_fn = cosine_exp
    elif dtwrc["dist_method"] == "cosine_neg_log":
        dist_fn = cosine_neg_log
    else:
        dist_fn = partial(distance.cdist, metric=dtwrc["dist_method"])
    log.warning(f"DTW distance function: {dtwrc['dist_method']}")

    # Define DTW configurations
    dtw_cfg = {
        "step_pattern": dtwrc["step_pattern"],
        "keep_internals": False,
        "distance_only": False if dtwrc["subsequence"] else True,
        "open_begin": True if dtwrc["subsequence"] else False,
        "open_end": True if dtwrc["subsequence"] else False,
    }
    log.warning(f"DTW configs: {dtw_cfg}")

    # Calculate matching scores
    results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
                        dtwrc["minmax_norm"],
                        dtw_cfg,
                    )
                )
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="DTW", ncols=0,
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
        str(expdir / "benchmark.stdlist.xml"),
        encoding="UTF-8",
        pretty_print=True,
    )


if __name__ == "__main__":
    from s3prl.utility.helper import override

    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--docs_dir")
    parser.add_argument("--config")
    parser.add_argument("--override", "-o")
    args = parser.parse_args()

    if args.config is None:
        files = glob.glob(str(Path(args.features_dir) / "config*"))
        files.sort(key=os.path.getmtime, reverse=True)
        config_file = files[-1]
    else:
        config_file = args.config
    assert os.path.isfile(config_file)
    docs_dir = args.features_dir if args.docs_dir is None else args.docs_dir

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if args.override is not None:
        message = override(args.override, args, config)
    for m in message:
        log.warning(m)

    dexpert = config["downstream_expert"]
    dtwrc = dexpert["dtwrc"]

    def load_pickle(pkl_dir: str, name: str):
        with open(Path(pkl_dir) / f"{name}.pkl", "rb") as file:
            return pickle.load(file)

    os.makedirs(args.output_dir, exist_ok=True)
    dtw_and_dump_tree(
        load_pickle(args.features_dir, "queries"),
        load_pickle(args.features_dir, "query_names"),
        load_pickle(docs_dir, "docs"),
        load_pickle(docs_dir, "doc_names"),
        dtwrc,
        args.output_dir,
    )
