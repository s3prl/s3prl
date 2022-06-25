from s3prl.dataset.base import AugmentedDynamicItemDataset


def test_replace_key():
    dataset = AugmentedDynamicItemDataset(
        {
            "1": dict(
                text="hello",
                path="cool",
            ),
            "2": dict(
                text="best",
                path="not cool",
            ),
        }
    )
    dataset.set_output_keys(["text"])
    dataset.replace_output_key("text", "token_id")
    assert dataset.pipeline.output_mapping["text"] == "token_id"
