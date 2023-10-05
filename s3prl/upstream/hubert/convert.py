import tempfile
from pathlib import Path
from typing import List

import torch

import s3prl
from s3prl.upstream.hubert.hubert_model import (
    HubertConfig,
    HubertModel,
    HubertPretrainingConfig,
)
from s3prl.upstream.utils import load_fairseq_ckpt, merge_with_parent
from s3prl.util.download import _urls_to_filepaths


def load_and_convert_fairseq_ckpt(fairseq_source: str, output_path: str = None):
    from fairseq.data.dictionary import Dictionary

    state, cfg = load_fairseq_ckpt(fairseq_source)

    dicts: List[Dictionary] = state["task_state"]["dictionaries"]
    symbols = [dictionary.symbols for dictionary in dicts]

    output_state = {
        "task_cfg": cfg["task"],
        "model_cfg": cfg["model"],
        "model_weight": state["model"],
        "dictionaries_symbols": symbols,
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(output_state, output_path)


def load_converted_model(ckpt: str):
    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in [
        "task_cfg",
        "model_cfg",
        "model_weight",
        "dictionaries_symbols",
    ]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )

    task_cfg = merge_with_parent(HubertPretrainingConfig, ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(HubertConfig, ckpt_state["model_cfg"])
    model = HubertModel(model_cfg, task_cfg, ckpt_state["dictionaries_symbols"])

    model.load_state_dict(ckpt_state["model_weight"])
    return model, task_cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fairseq_ckpt")
    parser.add_argument(
        "--output_dir", default=Path(s3prl.__file__).parent.parent / "converted_ckpts"
    )
    args = parser.parse_args()

    Path(args.output_dir).parent.mkdir(exist_ok=True, parents=True)
    load_and_convert_fairseq_ckpt(
        args.fairseq_ckpt, Path(args.output_dir) / f"{Path(args.fairseq_ckpt).stem}.pt"
    )
