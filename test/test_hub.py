import pytest
import shutil
import logging
import tempfile
import traceback
from pathlib import Path
from subprocess import check_call

import torch
import s3prl
from s3prl import hub
from s3prl.util.pseudo_data import get_pseudo_wavs
from s3prl.util.download import _urls_to_filepaths

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
EXTRACTED_GT_DIR = Path(s3prl.__file__).parent.parent / "sample_hidden_states"

# Expect the follow directory structure:
#
# -- s3prl/
# ---- s3prl/
# ------- hub.py
# ---- test/
# ---- sample_hidden_states/


def _prepare_sample_hidden_states():
    if not EXTRACTED_GT_DIR.is_dir():
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            tempdir.mkdir(exist_ok=True, parents=True)

            logger.info("Downloading extracted sample hidden states...")
            check_call("git lfs install".split(), cwd=tempdir)
            check_call(
                "git clone https://huggingface.co/datasets/s3prl/sample_hidden_states".split(),
                cwd=tempdir,
            )
            shutil.move(
                str(tempdir / "sample_hidden_states"), str(EXTRACTED_GT_DIR.parent)
            )
    else:
        logger.info(f"{EXTRACTED_GT_DIR} exists. Perform git pull...")
        check_call("git pull".split(), cwd=EXTRACTED_GT_DIR)


def _extract_feat(model: torch.nn.Module):
    wavs = get_pseudo_wavs()
    hidden_states = model(wavs)["hidden_states"]
    return hidden_states


def _all_hidden_states_same(hs1, hs2):
    for h1, h2 in zip(hs1, hs2):
        assert torch.allclose(h1, h2)


def _load_ground_truth(name: str):
    source = f"{EXTRACTED_GT_DIR}/{name}.pt"
    if source.startswith("http"):
        path = _urls_to_filepaths(source)
    else:
        path = source
    return torch.load(path)


def _compare_with_extracted(name: str):
    cls = getattr(hub, name)
    model_new = cls()
    model_new.eval()

    with torch.no_grad():
        hs_new = _extract_feat(model_new)
    hs_gt = _load_ground_truth(name)

    _all_hidden_states_same(hs_new, hs_gt)


def _test_model(name: str):
    """
    Test the upstream with the name: 'name' can successfully forward and backward
    """
    with torch.autograd.set_detect_anomaly(True):
        model = getattr(hub, name)()
        hs = _extract_feat(model)
        h_sum = 0
        for h in hs:
            h_sum = h_sum + h.sum()
        h_sum.backward()


"""
Test cases ensure that all upstreams are working and are same with pre-extracted features
"""


@pytest.mark.slow
def test_all_model():
    _prepare_sample_hidden_states()

    options = [
        name
        for name in hub.options(only_registered_ckpt=True)
        if (not name == "customized_upstream")
        and (
            not "mos" in name
        )  # mos models do not have hidden_states key. They only return a single mos score
        and (
            not "stft_mag" in name
        )  # stft_mag upstream must past the config file currently and is not so important. So, skip the test now
        and (
            not "pase" in name
        )  # pase_plus needs lots of dependencies and is difficult to be tested and is not very worthy today
        and (
            not name == "xls_r_1b"
        )  # skip due to too large model, too long download time
        and (
            not name == "xls_r_2b"
        )  # skip due to too large model, too long download time
    ]

    tracebacks = []
    for name in options:
        logger.info(f"Testing upstream: '{name}'")
        try:
            _compare_with_extracted(name)
            _test_model(name)

        except Exception as e:
            logger.error(f"{name}\n{traceback.format_exc()}")
            tb = traceback.format_exc()
            tracebacks.append((name, tb))

    if len(tracebacks) > 0:
        for name, tb in tracebacks:
            logger.error(f"Error in {name}:\n{tb}")
        logger.error(f"All failed models:\n{[name for name, _ in tracebacks]}")
        assert False


def test_one_model(upstream_name: str):
    if upstream_name is None:
        return

    _prepare_sample_hidden_states()
    _compare_with_extracted(upstream_name)
    _test_model(upstream_name)
