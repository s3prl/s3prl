from pathlib import Path

import torch

import s3prl
from s3prl.upstream.data2vec.data2vec_model import (
    Data2VecAudioConfig,
    Data2VecAudioModel,
)
from s3prl.upstream.utils import load_fairseq_ckpt, merge_with_parent
from s3prl.upstream.wav2vec2.wav2vec2_model import AudioPretrainingConfig


def load_and_convert_fairseq_ckpt(fairseq_source: str, output_path: str):
    state, cfg = load_fairseq_ckpt(fairseq_source)
    output_state = {
        "task_cfg": cfg["task"],
        "model_cfg": cfg["model"],
        "model_weight": state["model"],
    }

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(output_state, output_path)

    # make sure can load
    load_converted_model(output_path)


def load_converted_model(ckpt: str):
    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in ["task_cfg", "model_cfg", "model_weight"]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )

    task_cfg = merge_with_parent(AudioPretrainingConfig, ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(Data2VecAudioConfig, ckpt_state["model_cfg"])
    model = Data2VecAudioModel(model_cfg)

    model.remove_pretraining_modules()
    del ckpt_state["model_weight"]["_ema"]

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
