import logging

from s3prl import Container, field, Workspace
from s3prl.nn import S3PRLUpstreamDriver, UpstreamDownstreamModel
from s3prl.problem.base import Problem
from s3prl.problem.trainer import Trainer
from s3prl.util.configuration import default_cfg
from s3prl.util.seed import fix_random_seeds
from s3prl.dataset.hear_timestamp import HearTimestampDatapipe
from s3prl.sampler import FixedBatchSizeBatchSampler, GroupSameItemSampler

logger = logging.getLogger(__name__)


class HearTimestamp(Problem, Trainer):
    @default_cfg(
        workspace=field(
            "???",
            "\nWill put the following keys into this workspace:\n"
            "  'train_dataset', 'train_sampler', 'valid_dataset', 'valid_sampler', and 'task'",
            "str or Path or Workspace",
        ),
        corpus=dict(
            CLS=field(
                "???",
                "\nThe corpus class. You can add the **kwargs right below this CLS key",
                str,
            ),
            dataset_root=field(
                "???",
                "The root path of the corpus",
                str,
            ),
        ),
        train_datapipe=dict(
            CLS=field(
                HearTimestampDatapipe,
                "\nThe first datapipe class to be applied to the corpus. You can add the **kwargs right below this CLS key",
                str,
            ),
        ),
        train_sampler=dict(
            CLS=field(
                FixedBatchSizeBatchSampler,
                "\nThe batch sampler class. You can add the **kwargs right below this CLS key",
                str,
            ),
            batch_size=5,
        ),
        valid_datapipe=dict(
            CLS=field(
                HearTimestampDatapipe,
                "\nThe first datapipe class to be applied to the corpus. You can add the **kwargs right below this CLS key",
                str,
            ),
        ),
        valid_sampler=dict(
            CLS=GroupSameItemSampler,
            item_name="unchunked_id",
            item_order_name="chunk_index",
        ),
        test_datapipe=dict(
            CLS=field(
                HearTimestampDatapipe,
                "\nThe first datapipe class to be applied to the corpus. You can add the **kwargs right below this CLS key",
                str,
            ),
        ),
        test_sampler=dict(
            CLS=GroupSameItemSampler,
            item_name="unchunked_id",
            item_order_name="chunk_index",
        ),
        upstream=dict(
            CLS=field(
                S3PRLUpstreamDriver,
                "\nThe class of the upstream model following the specific interface. You can add the **kwargs right below this CLS key",
                str,
            ),
            name="hubert",
            feature_selection="hidden_states",
            freeze_upstream=field(
                True,
                "Set the entire upstream model's requires_grad to False, or else, leave it alone",
            ),
            normalize=field(
                False, "Apply layer-norm to upstream model's each layer hidden_state"
            ),
            weighted_sum=field(
                True,
                "If True, apply weighted-sum on the selected layers; If False, take the final layer.\n"
                "For the selected layers, see the 'layer_selections' option",
            ),
            layer_selections=field(
                None,
                "If None, select all layers; Or, select the subset layers defined by this option",
            ),
        ),
        downstream=dict(
            CLS=field(
                "???",
                "\nThe downstream model class for each task. You can add the **kwargs right below this CLS key",
                str,
            ),
        ),
        task=dict(
            CLS=field(
                "???",
                "\nThe task class defining what to do for each train/valid/test step in the train/valid/test dataloader loop"
                "\nYou can add the **kwargs right below this CLS key",
                str,
            ),
        ),
    )
    @classmethod
    def setup(cls, **cfg) -> Container:
        cfg = Container(cfg)
        workspace = Workspace(cfg.workspace)
        fix_random_seeds()

        upstream = cfg.upstream()
        stats = Container(
            feat_frame_shift=upstream.downsample_rate,
        )

        logger.info("Preparing corpus")
        train_data, valid_data, test_data, corpus_stats = cfg.corpus().split(3)
        stats = corpus_stats.add(stats)

        logger.info("Preparing train data")
        train_dataset = cfg.train_datapipe(**stats)(train_data, **stats)
        train_sampler = cfg.train_sampler(train_dataset)
        stats.override(train_dataset.all_tools())
        workspace.environ.update(stats)

        logger.info("Preparing valid data")
        valid_dataset = cfg.valid_datapipe(**dict(workspace.environ))(
            valid_data, **dict(workspace.environ)
        )
        valid_sampler = cfg.valid_sampler(valid_dataset)

        logger.info("Preparing test data")
        test_dataset = cfg.test_datapipe(**dict(workspace.environ))(
            test_data, **dict(workspace.environ)
        )
        test_sampler = cfg.test_sampler(test_dataset)

        logger.info("Preparing model and task")
        downstream = cfg.downstream(upstream.output_size, **dict(workspace.environ))
        model = UpstreamDownstreamModel(upstream, downstream)
        task = cfg.task(model, **dict(workspace.environ))

        workspace["train_data"] = train_data
        workspace["valid_data"] = valid_data
        workspace["test_data"] = test_data
        workspace["train_dataset"] = train_dataset
        workspace["train_sampler"] = train_sampler
        workspace["valid_dataset"] = valid_dataset
        workspace["valid_sampler"] = valid_sampler
        workspace["test_dataset"] = test_dataset
        workspace["test_sampler"] = test_sampler
        workspace["task"] = task
