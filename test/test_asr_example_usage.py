import pytest
from dotenv import dotenv_values

import torch

from s3prl.problem import SuperbASR
from s3prl.nn import UpstreamDownstreamModel
from s3prl.dataset.base import DataLoader


@pytest.mark.corpus
def test_asr_example_usage():
    LIBRISPEECH = dotenv_values()["LibriSpeech"]
    default_cfg = SuperbASR.setup.default_cfg
    train_data, valid_data, test_data = default_cfg.corpus(dataset_root=LIBRISPEECH).slice(3)

    train_dataset = default_cfg.train_datapipe()(train_data)
    stats = train_dataset.all_tools()

    valid_dataset = default_cfg.valid_datapipe()(valid_data, **stats)
    test_dataset = default_cfg.test_datapipe()(test_data, **stats)

    upstream = default_cfg.upstream(name="hubert")
    downstream = default_cfg.downstream(input_size=upstream.output_size, **stats)
    model = UpstreamDownstreamModel(upstream, downstream)
    task = default_cfg.task(model=model, **stats)

    train_sampler = default_cfg.train_sampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, train_sampler)
    train_iter = iter(train_dataloader)

    loss = task.train_step(**next(train_iter)).loss
    assert isinstance(loss, torch.Tensor)
