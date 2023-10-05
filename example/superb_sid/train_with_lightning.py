import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from s3prl.nn import S3PRLUpstream, UpstreamDownstreamModel
from s3prl.superb import sid as problem
from s3prl.wrapper import LightningModuleSimpleWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("voxceleb1", help="The root directory of VoxCeleb1")
    parser.add_argument("save_to", help="The directory to save checkpoint")
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--eval_step", type=int, default=5000)
    parser.add_argument("--save_step", type=int, default=100)
    parser.add_argument(
        "--not_resume",
        action="store_true",
        help="Don't resume from the last checkpoint",
    )

    # for debugging
    parser.add_argument("--limit_train_batches", type=int)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    voxceleb1 = Path(args.voxceleb1)
    save_to = Path(args.save_to)
    save_to.mkdir(exist_ok=True, parents=True)

    logger.info("Preparing preprocessor")
    preprocessor = problem.Preprocessor(voxceleb1)

    logger.info("Preparing train dataloader")
    train_dataset = problem.TrainDataset(**preprocessor.train_data())
    train_dataloader = train_dataset.to_dataloader(
        batch_size=8,
        num_workers=6,
        shuffle=True,
    )

    logger.info("Preparing valid dataloader")
    valid_dataset = problem.ValidDataset(
        **preprocessor.valid_data(),
        **train_dataset.statistics(),
    )
    valid_dataloader = valid_dataset.to_dataloader(batch_size=8, num_workers=6)

    logger.info("Preparing test dataloader")
    test_dataset = problem.TestDataset(
        **preprocessor.test_data(),
        **train_dataset.statistics(),
    )
    test_dataloader = test_dataset.to_dataloader(batch_size=8, num_workers=6)

    valid_dataset.save_checkpoint(save_to / "valid_dataset.ckpt")
    test_dataset.save_checkpoint(save_to / "test_dataset.ckpt")

    upstream = S3PRLUpstream("apc")
    downstream = problem.DownstreamModel(
        upstream.output_size, len(preprocessor.statistics().category)
    )
    model = UpstreamDownstreamModel(upstream, downstream)
    task = problem.Task(model, preprocessor.statistics().category)

    optimizer = optim.Adam(task.parameters(), lr=1e-3)
    lightning_task = LightningModuleSimpleWrapper(task, optimizer)

    # The above is the usage of our library

    # The below is pytorch-lightning specific usage, which can be very simple
    # or very sophisticated, depending on how much you want to customized your
    # training loop

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_to),
        filename="superb-sid-{step:02d}-{valid_0_accuracy:.2f}",
        monitor="valid_0_accuracy",  # since might have multiple valid dataloaders
        save_last=True,
        save_top_k=3,  # top 3 best ckpt on valid
        mode="max",  # higher, better
        every_n_train_steps=args.save_step,
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        gpus=1,
        max_steps=args.total_steps,
        log_every_n_steps=args.log_step,
        val_check_interval=args.eval_step,
        limit_val_batches=args.limit_val_batches or 1.0,
        limit_train_batches=args.limit_train_batches or 1.0,
        fast_dev_run=args.fast_dev_run,
    )

    last_ckpt = save_to / "last.ckpt"
    if args.not_resume or not last_ckpt.is_file():
        last_ckpt = None

    trainer.fit(
        lightning_task,
        train_dataloader,
        val_dataloaders=[valid_dataloader, test_dataloader],
        ckpt_path=last_ckpt,
    )


if __name__ == "__main__":
    main()
