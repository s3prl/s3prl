import logging
import tempfile
from pathlib import Path

import torch
from dotenv import dotenv_values
from torch.utils.data import DataLoader

from s3prl import Object, Output
from s3prl.nn import S3PRLUpstream, UpstreamDownstreamModel
from s3prl.superb import example
from s3prl.superb import sid as problem

TRIAL_STEP = 5
BATCH_SIZE = 4
TIMESTAMPS = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)


def test_superb_sid(helpers):
    env = dotenv_values()
    voxceleb1 = Path(env["VoxCeleb1"])
    if voxceleb1.is_dir():
        logger.info("Use SID preprocessor")
        preprocessor = problem.Preprocessor(voxceleb1)
    else:
        logger.info("Use pseudo preprocessor due to incorrect VoxCeleb1 path in .env")
        preprocessor = example.Preprocessor(
            "your_dataset", train_ratio=0.6, valid_ratio=0.2
        )

    train_dataset = problem.TrainDataset(**preprocessor.train_data())
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, collate_fn=train_dataset.collate_fn, num_workers=0
    )

    valid_dataset = problem.ValidDataset(
        **preprocessor.valid_data(),
        **train_dataset.statistics(),
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, collate_fn=valid_dataset.collate_fn, num_workers=0
    )

    test_dataset = problem.TestDataset(
        **preprocessor.test_data(),
        **train_dataset.statistics(),
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn, num_workers=0
    )

    upstream = S3PRLUpstream("apc")
    downstream = problem.DownstreamModel(
        upstream.output_size, len(preprocessor.statistics().category)
    )
    model = UpstreamDownstreamModel(upstream, downstream)
    task = problem.Task(model, preprocessor.statistics().category)
    task = task.to(device)

    test_suit = (
        (train_dataloader, task.train_step, task.train_reduction),
        (valid_dataloader, task.valid_step, task.valid_reduction),
        (test_dataloader, task.test_step, task.test_reduction),
    )

    for dataloader, step_fn, reduction_fn in test_suit:
        batch_results = []
        for idx, batch in enumerate(dataloader):
            assert isinstance(batch, Output)
            batch = batch.to(device)

            result = step_fn(**batch)
            assert isinstance(result, Output)
            result.loss.backward()

            cacheable_result = result.cacheable()
            batch_results.append(cacheable_result)

            if idx > TRIAL_STEP:
                break

        logs = reduction_fn(batch_results).logs
        assert "loss" in logs

    # Test reload dataset
    with tempfile.NamedTemporaryFile() as file:
        test_dataset.save_checkpoint(file.name)
        test_dataset_reload = Object.load_checkpoint(file.name)

    tensor1 = helpers.get_single_tensor(test_dataset[0])
    tensor2 = helpers.get_single_tensor(test_dataset_reload[0])
    assert torch.allclose(tensor1, tensor2)

    # Test reload task module
    pseudo_input = torch.randn(BATCH_SIZE, TIMESTAMPS, task.input_size).to(device)
    pseudo_input_len = (torch.ones(BATCH_SIZE).long() * TIMESTAMPS).to(device)
    assert helpers.validate_module(
        task,
        pseudo_input,
        pseudo_input_len,
        device=device,
    )

    # Test reload model
    model = task.model
    pseudo_input = torch.randn(BATCH_SIZE, TIMESTAMPS, model.input_size).to(device)
    pseudo_input_len = (torch.ones(BATCH_SIZE).long() * TIMESTAMPS).to(device)
    assert helpers.validate_module(
        model,
        pseudo_input,
        pseudo_input_len,
        device=device,
    )

    # Test reload a single submodule
    downstream = task.model.downstream
    pseudo_input = torch.randn(BATCH_SIZE, TIMESTAMPS, downstream.input_size).to(device)
    pseudo_input_len = (torch.ones(BATCH_SIZE).long() * TIMESTAMPS).to(device)
    assert helpers.validate_module(
        downstream,
        pseudo_input,
        pseudo_input_len,
        device=device,
    )
