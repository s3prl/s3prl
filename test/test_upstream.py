import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from subprocess import check_call

import pytest
import torch
from filelock import FileLock

from s3prl.nn import Featurizer, S3PRLUpstream
from s3prl.util.download import _urls_to_filepaths
from s3prl.util.pseudo_data import get_pseudo_wavs

logger = logging.getLogger(__name__)

TEST_MORE_ITER = 2
TRAIN_MORE_ITER = 5
SAMPLE_RATE = 16000
ATOL = 0.01
MAX_LENGTH_DIFF = 3
EXTRA_SHORT_SEC = 0.001
EXTRACTED_GT_DIR = Path(__file__).parent.parent / "sample_hidden_states"

# Expect the following directory structure:
#
# -- s3prl/  (repository root)
# ---- s3prl/  (package root)
# ---- test/
# ------- test_upstream.py
# ---- sample_hidden_states/


def _prepare_sample_hidden_states():
    lock_file = Path(__file__).parent.parent / "sample_hidden_states.lock"
    with FileLock(str(lock_file)):

        # NOTE: home variable is necessary for git lfs to work
        env = dict(os.environ)
        if not "HOME" in env:
            env["HOME"] = Path.home()

        if not EXTRACTED_GT_DIR.is_dir():
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                tempdir.mkdir(exist_ok=True, parents=True)

                logger.info("Downloading extracted sample hidden states...")
                check_call("git lfs install".split(), cwd=tempdir, env=env)
                check_call(
                    "git clone https://huggingface.co/datasets/s3prl/sample_hidden_states".split(),
                    cwd=tempdir,
                    env=env,
                )
                shutil.move(
                    str(tempdir / "sample_hidden_states"), str(EXTRACTED_GT_DIR.parent)
                )
        else:
            logger.info(f"{EXTRACTED_GT_DIR} exists. Perform git pull...")
            check_call("git pull".split(), cwd=EXTRACTED_GT_DIR, env=env)

    try:
        lock_file.unlink()
    except FileNotFoundError:
        pass


def _extract_feat(
    model: S3PRLUpstream,
    seed: int = 0,
    **pseudo_wavs_args,
):
    wavs, wavs_len = get_pseudo_wavs(seed=seed, padded=True, **pseudo_wavs_args)
    all_hs, all_lens = model(wavs, wavs_len)
    return all_hs


def _all_hidden_states_same(hs1, hs2):
    for h1, h2 in zip(hs1, hs2):
        if h1.size(1) != h2.size(1):
            length_diff = abs(h1.size(1) - h2.size(1))
            assert length_diff <= MAX_LENGTH_DIFF, f"{length_diff} > {MAX_LENGTH_DIFF}"
            min_seqlen = min(h1.size(1), h2.size(1))
            h1 = h1[:, :min_seqlen, :]
            h2 = h2[:, :min_seqlen, :]
            assert torch.allclose(h1, h2, atol=ATOL)


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


def _test_forward_backward(name: str, **pseudo_wavs_args):
    """
    Test the upstream with the name: 'name' can successfully forward and backward
    """
    with torch.autograd.set_detect_anomaly(True):
        model = S3PRLUpstream(name)
        hs = _extract_feat(model, **pseudo_wavs_args)
        h_sum = 0
        for h in hs:
            h_sum = h_sum + h.sum()
        h_sum.backward()


def _filter_options(options: list):
    options = [
        name
        for name in options
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
    return options


"""
Test cases ensure that all upstreams are working and are same with pre-extracted features
"""


@pytest.mark.upstream
@pytest.mark.parametrize(
    "name",
    [
        "wav2vec2",
        "wavlm",
        "hubert",
    ],
)
def test_common_upstream(name):
    _prepare_sample_hidden_states()
    _compare_with_extracted(name)
    _test_forward_backward(
        name, min_secs=EXTRA_SHORT_SEC, max_secs=EXTRA_SHORT_SEC, n=1
    )
    _test_forward_backward(
        name, min_secs=EXTRA_SHORT_SEC, max_secs=EXTRA_SHORT_SEC, n=2
    )
    _test_forward_backward(name, min_secs=EXTRA_SHORT_SEC, max_secs=1, n=3)


@pytest.mark.upstream
@pytest.mark.slow
def test_upstream_with_extracted(upstream_names: str):
    _prepare_sample_hidden_states()

    if upstream_names is not None:
        options = upstream_names.split(",")
    else:
        options = S3PRLUpstream.available_names(only_registered_ckpt=True)
        options = _filter_options(options)
        options = sorted(options)

    tracebacks = []
    for name in options:
        logger.info(f"Testing upstream: '{name}'")
        try:
            _compare_with_extracted(name)

        except Exception as e:
            logger.error(f"{name}\n{traceback.format_exc()}")
            tb = traceback.format_exc()
            tracebacks.append((name, tb))

    if len(tracebacks) > 0:
        for name, tb in tracebacks:
            logger.error(f"Error in {name}:\n{tb}")
        logger.error(f"All failed models:\n{[name for name, _ in tracebacks]}")
        assert False


@pytest.mark.upstream
@pytest.mark.slow
def test_upstream_forward_backward(upstream_names: str):
    if upstream_names is not None:
        options = upstream_names.split(",")
    else:
        options = S3PRLUpstream.available_names(only_registered_ckpt=True)
        options = _filter_options(options)
        options = sorted(options)
        options = reversed(options)

    tracebacks = []
    for name in options:
        logger.info(f"Testing upstream: '{name}'")
        try:
            _test_forward_backward(name)

        except Exception as e:
            logger.error(f"{name}\n{traceback.format_exc()}")
            tb = traceback.format_exc()
            tracebacks.append((name, tb))

    if len(tracebacks) > 0:
        for name, tb in tracebacks:
            logger.error(f"Error in {name}:\n{tb}")
        logger.error(f"All failed models:\n{[name for name, _ in tracebacks]}")
        assert False


@pytest.mark.upstream
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


@pytest.mark.upstream
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
