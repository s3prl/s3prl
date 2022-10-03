import argparse
import glob
import os
from pathlib import Path
from subprocess import check_call

import torch

import s3prl
from s3prl.upstream.roberta.dictionary import Dictionary
from s3prl.upstream.roberta.roberta_model import (
    MaskedLMConfig,
    RobertaEncoder,
    RobertaModel,
)
from s3prl.upstream.utils import load_fairseq_ckpt, merge_with_parent
from s3prl.util.download import _urls_to_filepaths


def load_and_convert_fairseq_ckpt(fairseq_source: str, output_path: str):
    """
    Args:
        fairseq_source (str): either URL for the tar file or the untared directory path
        output_path (str): converted checkpoint path
    """
    if fairseq_source.startswith("http"):
        tar_file = _urls_to_filepaths(fairseq_source)
        tar_dir = Path(tar_file).parent / "vq_wav2vec_kmeans_roberta/"
        tar_dir.mkdir(exist_ok=True, parents=True)
        check_call(cwd=f"tar -xf {tar_file} -C {tar_dir}".split(), shell=True)
    else:
        fairseq_source = Path(fairseq_source)
        assert fairseq_source.is_dir()
        tar_dir = fairseq_source

    pt_files = glob.glob(os.path.join(tar_dir, "*.pt"))
    assert len(pt_files) == 1
    pt_file = pt_files[0]

    state, cfg = load_fairseq_ckpt(
        tar_dir / Path(pt_file).name,
        bpe="gpt2",
        load_checkpoint_heads=True,
    )
    assert isinstance(
        cfg["model"], argparse.Namespace
    ), "RoBERTa pre-training does not have dataclass config and only accepts Namespace"

    with (tar_dir / "dict.txt").open() as f:
        text_dictionary: str = f.read()

    output_state = {
        "task_cfg": cfg["task"],
        "model_cfg": cfg["model"],
        "model_weight": state["model"],
        "text_dictionary": text_dictionary,
    }
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(output_state, output_path)

    load_converted_model(output_path)


def load_converted_model(ckpt: str):
    import tempfile

    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in ["task_cfg", "model_cfg", "model_weight", "text_dictionary"]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )

    with tempfile.NamedTemporaryFile() as f:
        with open(f.name, "w") as f_handle:
            f_handle.write(ckpt_state["text_dictionary"])
        dictionary = Dictionary.load(f.name)
        dictionary.add_symbol("<mask>")

    model_cfg = ckpt_state["model_cfg"]
    assert isinstance(
        model_cfg, argparse.Namespace
    ), "RoBERTa pre-training does not have dataclass config and only accepts Namespace"

    encoder = RobertaEncoder(model_cfg, dictionary)
    model = RobertaModel(model_cfg, encoder)
    model.load_state_dict(ckpt_state["model_weight"])

    task_cfg = merge_with_parent(MaskedLMConfig, ckpt_state["task_cfg"])
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
