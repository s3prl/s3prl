from collections import Counter
from s3prl.dataset.base import AugmentedDynamicItemDataset
from s3prl.dataset.fewshot import BalancedRatioSampler

def test_balanced_ratio_sampler():
    data = {
        "1": dict(
            speaker="meow",
        ),
        "2": dict(
            speaker="meow",
        ),
        "3": dict(
            speaker="best",
        ),
        "4": dict(
            speaker="good",
        ),
        "5": dict(
            speaker="good",
        ),
        "6": dict(
            speaker="good",
        ),
        "7": dict(
            speaker="good",
        ),
        "8": dict(
            speaker="good",
        ),
        "9": dict(
            speaker="good",
        ),
        "10": dict(
            speaker="good",
        ),
    }
    dataset = AugmentedDynamicItemDataset(data)
    dataset: AugmentedDynamicItemDataset = BalancedRatioSampler(target_name="speaker", ratio=0.5)(dataset)
    with dataset.output_keys_as(["speaker"]):
        speakers = [item["speaker"] for item in dataset]
        speakers = Counter(speakers)
    assert speakers["meow"] == 2
    assert speakers["good"] == 2
    assert speakers["best"] == 1
