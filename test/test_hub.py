import pytest
import logging
import traceback

import torch
from s3prl import hub
from s3prl.util.pseudo_data import get_pseudo_wavs
from s3prl.util.download import _urls_to_filepaths

logger = logging.getLogger()

SAMPLE_RATE = 16000
EXTRACTED_GROUND_TRUTH_URL = "../sample_hidden_states/"


NAMES_WITH_LEGACY = [
    "wav2vec2_base_960",
    "wav2vec2_large_960",
    "wav2vec2_large_ll60k",
    "wav2vec2_large_lv60_cv_swbd_fsh",
    "xlsr_53",
    "xls_r_300m",
    "wav2vec2_conformer_relpos",
    "wav2vec2_conformer_rope",
    "hubert_base",
    "hubert_large_ll60k",
    "hubert_base_robust_mgr",
    "wav2vec_large",
    "vq_wav2vec_gumbel",
    "vq_wav2vec_kmeans",
    "distilhubert_base",
    "decoar2",
    "data2vec_base_960",
    "data2vec_large_ll60k",
    "discretebert",
]


def _extract_feat(model: torch.nn.Module):
    wavs = get_pseudo_wavs()
    hidden_states = model(wavs)["hidden_states"]
    return hidden_states


def _all_hidden_states_same(hs1, hs2):
    for h1, h2 in zip(hs1, hs2):
        assert torch.allclose(h1, h2)


def _load_ground_truth(name: str):
    source = f"{EXTRACTED_GROUND_TRUTH_URL}/{name}.pt"
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


def _compare_with_fairseq(name: str):
    cls = getattr(hub, name)
    model_old = cls(legacy=True)
    model_old.eval()
    model_new = cls(legacy=False)
    model_new.eval()

    with torch.no_grad():
        hs_old = _extract_feat(model_old)
        hs_new = _extract_feat(model_new)

    _all_hidden_states_same(hs_old, hs_new)


@pytest.mark.slow
def test_one_hub(upstream_name):
    if upstream_name is not None:
        _compare_with_extracted(upstream_name)


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES_WITH_LEGACY)
def test_all_hub(name: str):
    _compare_with_extracted(name)


@pytest.mark.fairseq
def test_one_hub_fairseq(upstream_name):
    if upstream_name is not None:
        _compare_with_fairseq(upstream_name)


@pytest.mark.fairseq
@pytest.mark.parametrize("name", NAMES_WITH_LEGACY)
def test_all_hub_fairseq(name: str):
    _compare_with_fairseq(name)


def _test_model(name: str):
    if name is not None:
        with torch.autograd.set_detect_anomaly(True):
            try:
                logger.info(f"Testing upstream: '{name}'")
                model = getattr(hub, name)()
                hs = _extract_feat(model)
                h_sum = 0
                for h in hs:
                    h_sum = h_sum + h.sum()
                h_sum.backward()
                return []
            except Exception as e:
                logger.error(f"{name}\n{traceback.format_exc()}")
                return [(name, traceback.format_exc())]


def test_one_model(upstream_name: str):
    assert len(_test_model(upstream_name)) == 0


@pytest.mark.slow
def test_all_model():
    options = [
        name
        for name in hub.available_options()
        if (not "local" in name)
        and (not "url" in name)
        and (not "custom" in name)
        and (
            not "mos" in name
        )  # mos models do not have hidden_states key. They only return a single mos score
    ]
    options.remove(
        "stft_mag"
    )  # stft_mag upstream must past the config file currently and is not so important. So, skip the test now
    options.remove(
        "pase_plus"
    )  # pase_plus needs lots of dependencies and is difficult to be tested and is not very worthy today

    # FIXME: (Leo) These two upstreams should be tested but ignored
    # for now due to high memory cost
    options.remove("xls_r_1b")
    options.remove("xls_r_2b")

    tracebacks = []
    for name in options:
        tracebacks.extend(_test_model(name))

    if len(tracebacks) > 0:
        for name, tb in tracebacks:
            logger.error(f"Error in {name}:\n{tb}")
        logger.error(f"All failed models:\n{[name for name, _ in tracebacks]}")
        assert False
