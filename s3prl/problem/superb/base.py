import logging

import torch.nn as nn

from s3prl import Container, field
from s3prl.dataset.base import AugmentedDynamicItemDataset
from s3prl.nn import S3PRLUpstream, UpstreamDownstreamModel
from s3prl.problem.base import Problem
from s3prl.problem.trainer import Trainer
from s3prl.util import workspace
from s3prl.base import Logs
from s3prl.dataset.base import DataLoader
from s3prl.util.configuration import default_cfg
from s3prl.util.workspace import Workspace, as_type

logger = logging.getLogger(__name__)


class SuperbProblem(Problem, Trainer):
    @default_cfg(
        workspace=field(
            "???",
            "\nWill put the following keys into this workspace:\n"
            "  'train_dataset', 'train_sampler', 'valid_dataset', 'valid_sampler', and 'task'",
            "str or Path or Workspace",
        ),
        corpus=dict(
            _cls=field(
                "???",
                "\nThe corpus class. You can add the **kwargs right below this _cls key",
                str,
            ),
            dataset_root=field("???", "The root path of the corpus", str),
        ),
        train_datapipe=dict(
            _cls=field(
                "???",
                "\nThe datapipe class to be applied to the corpus. You can add the **kwargs right below this _cls key",
                str,
            ),
        ),
        train_sampler=dict(
            _cls=field(
                "???",
                "\nThe batch sampler class. You can add the **kwargs right below this _cls key",
                str,
            ),
        ),
        valid_datapipe=dict(
            _cls=field(
                "???",
                "\nThe datapipe class to be applied to the corpus. You can add the **kwargs right below this _cls key",
                str,
            ),
        ),
        valid_sampler=dict(
            _cls=field(
                "???",
                "\nThe batch sampler class. You can add the **kwargs right below this _cls key",
                str,
            ),
        ),
        test_datapipe=dict(
            _cls=field(
                "???",
                "\nThe datapipe class to be applied to the corpus. You can add the **kwargs right below this _cls key",
                str,
            ),
        ),
        test_sampler=dict(
            _cls=field(
                "???",
                "\nThe batch sampler class. You can add the **kwargs right below this _cls key",
                str,
            ),
        ),
        upstream=dict(
            _cls=field(
                S3PRLUpstream,
                "\nThe class of the upstream model following the specific interface. You can add the **kwargs right below this _cls key",
                str,
            ),
            name="???",
            feature_selection="hidden_states",
        ),
        downstream=dict(
            _cls=field(
                "???",
                "\nThe downstream model class for each task. You can add the **kwargs right below this _cls key",
                str,
            ),
        ),
        task=dict(
            _cls=field(
                "???",
                "\nThe task class defining what to do for each train/valid/test step in the train/valid/test dataloader loop"
                "\nYou can add the **kwargs right below this _cls key",
                str,
            ),
        ),
    )
    @classmethod
    def setup_problem(cls, **cfg):
        cfg = Container(cfg)
        workspace = Workspace(cfg.workspace)

        if not isinstance(cfg.upstream, nn.Module):
            upstream = cfg.upstream._cls(**cfg.upstream.public())
        else:
            upstream = cfg.upstream

        stats = Container(upstream_rate=upstream.downsample_rate)

        logger.info("Preparing corpus")
        corpus = cfg.corpus._cls(**cfg.corpus.public())
        train_data, valid_data, test_data, corpus_stats = corpus().split(3)
        stats.add(corpus_stats)

        logger.info("Preparing train data")
        train_dataset = AugmentedDynamicItemDataset(train_data, tools=stats)
        train_dataset = cfg.train_datapipe._cls(**cfg.train_datapipe.public())(
            train_dataset
        )
        train_sampler = cfg.train_sampler._cls(
            train_dataset, **cfg.train_sampler.public()
        )
        stats.add(train_dataset.all_tools())

        logger.info("Preparing valid data")
        valid_dataset = AugmentedDynamicItemDataset(valid_data, tools=stats)
        valid_dataset = cfg.valid_datapipe._cls(**cfg.valid_datapipe.public())(
            valid_dataset
        )
        valid_sampler = cfg.valid_sampler._cls(
            valid_dataset, **cfg.valid_sampler.public()
        )

        logger.info("Preparing test data")
        test_dataset = AugmentedDynamicItemDataset(test_data, tools=stats)
        test_dataset = cfg.test_datapipe._cls(**cfg.test_datapipe.public())(
            test_dataset
        )
        test_sampler = cfg.test_sampler._cls(test_dataset, **cfg.test_sampler.public())

        logger.info("Preparing model and task")
        downstream = cfg.downstream._cls(
            upstream.output_size,
            **stats,
            **cfg.downstream.public(),
        )
        model = UpstreamDownstreamModel(upstream, downstream)
        task = cfg.task._cls(model, **stats, **cfg.task.public())

        workspace["train_dataset"] = train_dataset
        workspace["train_sampler"] = train_sampler
        workspace["valid_dataset"] = valid_dataset
        workspace["valid_sampler"] = valid_sampler
        workspace["test_dataset"] = test_dataset
        workspace["test_sampler"] = test_sampler
        workspace["task"] = task
        workspace.environ.update(stats)
