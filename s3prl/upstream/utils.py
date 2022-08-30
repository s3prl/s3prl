import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass, is_dataclass

import torch

from s3prl.util.download import _urls_to_filepaths
from s3prl.util.pseudo_data import get_pseudo_wavs

logger = logging.getLogger(__name__)


def load_fairseq_ckpt(source: str, **override):
    from fairseq.checkpoint_utils import load_checkpoint_to_cpu
    from omegaconf import OmegaConf

    source = str(source)
    if source.startswith("http"):
        fairseq_path = _urls_to_filepaths(source)
    else:
        fairseq_path = source

    state = load_checkpoint_to_cpu(fairseq_path, arg_overrides=override)
    cfg = OmegaConf.to_container(state["cfg"])

    assert type(cfg) == dict
    return state, cfg


def merge_with_parent(dc: dataclass, cfg: dict):

    assert is_dataclass(dc)
    assert type(cfg) == dict
    cfg = deepcopy(cfg)

    def fix_cfg(cfg):
        target_keys = set(dc.__dataclass_fields__.keys())
        for k in list(cfg.keys()):
            if k not in target_keys:
                del cfg[k]

    fix_cfg(cfg)
    assert len(cfg) > 0
    return dc(**cfg)


def extract_hidden_states(model):
    model.eval()
    with torch.no_grad():
        return model(get_pseudo_wavs())["hidden_states"]


def are_same_models(model1, model2):
    hs1 = extract_hidden_states(model1)
    hs2 = extract_hidden_states(model2)
    for h1, h2 in zip(hs1, hs2):
        assert torch.allclose(h1, h2)


def models_all_close(*models):
    assert len(models) > 1
    for model in models[1:]:
        are_same_models(models[0], model)
