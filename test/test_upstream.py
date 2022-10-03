import pytest
import shutil
import logging
import tempfile
import traceback
from pathlib import Path
from subprocess import check_call

import torch
from s3prl.nn import S3PRLUpstream, Featurizer
from s3prl.util.pseudo_data import get_pseudo_wavs
from s3prl.util.download import _urls_to_filepaths

logger = logging.getLogger(__name__)

TEST_MORE_ITER = 2
TRAIN_MORE_ITER = 5
SAMPLE_RATE = 16000
EXTRACTED_GT_DIR = Path(__file__).parent.parent / "sample_hidden_states"

# Expect the following directory structure:
#
# -- s3prl/  (repository root)
# ---- s3prl/  (package root)
# ---- test/
# ------- test_upstream.py
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


def _extract_feat(model: S3PRLUpstream, seed: int = 0):
    wavs, wavs_len = get_pseudo_wavs(seed=seed, padded=True)
    all_hs, all_lens = model(wavs, wavs_len)
    return all_hs


def _all_hidden_states_same(hs1, hs2):
    for h1, h2 in zip(hs1, hs2):
        if h1.size(1) != h2.size(1):
            min_seqlen = min(h1.size(1), h2.size(1))
            h1 = h1[:, :min_seqlen, :]
            h2 = h2[:, :min_seqlen, :]
        assert torch.allclose(h1, h2)


def _load_ground_truth(name: str):
    source = f"{EXTRACTED_GT_DIR}/{name}.pt"
    if source.startswith("http"):
        path = _urls_to_filepaths(source)
    else:
        path = source
    return torch.load(path)


def _compare_with_extracted(name: str):
    model = S3PRLUpstream(name)
    model.eval()

    with torch.no_grad():
        hs = _extract_feat(model)
        hs_gt = _load_ground_truth(name)

        _all_hidden_states_same(hs, hs_gt)

        for i in range(TEST_MORE_ITER):
            more_hs = _extract_feat(model)
            for h1, h2 in zip(hs, more_hs):
                assert torch.allclose(
                    h1, h2
                ), "should have deterministic representation in eval mode"

        for i in range(TEST_MORE_ITER):
            more_hs = _extract_feat(model, seed=i + 1)
            assert len(hs) == len(
                more_hs
            ), "should have deterministic num_layer in eval mode"

    model.train()
    for i in range(TRAIN_MORE_ITER):
        more_hs = _extract_feat(model, seed=i + 1)
        assert len(hs) == len(
            more_hs
        ), "should have deterministic num_layer in train mode"


def _test_model(name: str):
    """
    Test the upstream with the name: 'name' can successfully forward and backward
    """
    with torch.autograd.set_detect_anomaly(True):
        model = S3PRLUpstream(name)
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
        for name in S3PRLUpstream.available_names(only_registered_ckpt=True)
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


@pytest.mark.parametrize("name", ["lighthubert", "vggish", "mae_ast_frame"])
def test_forward_backward(name: str):
    _test_model(name)


@pytest.mark.extra_dependency
def test_ssast():
    _test_model("ssast_frame_base")


@pytest.mark.extra_dependency
def test_ast():
    _test_model("ast")


@pytest.mark.parametrize("layer_selections", [None, [0, 4, 9]])
@pytest.mark.parametrize("normalize", [False, True])
def test_featurizer(layer_selections, normalize):
    model = S3PRLUpstream("hubert")
    featurizer = Featurizer(
        model, layer_selections=layer_selections, normalize=normalize
    )

    wavs, wavs_len = get_pseudo_wavs(padded=True)
    all_hs, all_lens = model(wavs, wavs_len)
    hs, hs_len = featurizer(all_hs, all_lens)

    assert isinstance(hs, torch.FloatTensor)
    assert isinstance(hs_len, torch.LongTensor)


def test_upstream_properties():
    model = S3PRLUpstream("hubert")
    featurizer = Featurizer(model)
    assert isinstance(model.hidden_sizes, (tuple, list)) and isinstance(
        model.hidden_sizes[0], int
    )
    assert isinstance(model.downsample_rates, (tuple, list)) and isinstance(
        model.downsample_rates[0], int
    )
    assert isinstance(featurizer.output_size, int)
    assert isinstance(featurizer.downsample_rate, int)
