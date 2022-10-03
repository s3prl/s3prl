import logging
from collections import defaultdict

from s3prl import Container, field, Workspace
from s3prl.nn import S3PRLUpstreamDriver, UpstreamDownstreamModel
from s3prl.problem.base import Problem
from s3prl.problem.trainer import Trainer
from s3prl.util.configuration import default_cfg
from s3prl.util.seed import fix_random_seeds
from s3prl.dataset.utterance_classification_pipe import HearScenePipe
from s3prl.sampler import FixedBatchSizeBatchSampler
from s3prl.task.utterance_classification_task import UtteranceClassificationTask
from s3prl.nn.hear import HearFullyConnectedPrediction
from s3prl.task.scene_prediction import ScenePredictionTask

logger = logging.getLogger(__name__)


class HearScene(Problem, Trainer):
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
                HearScenePipe,
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
            batch_size="???",
        ),
        valid_datapipe=dict(
            CLS=field(
                HearScenePipe,
                "\nThe first datapipe class to be applied to the corpus. You can add the **kwargs right below this CLS key",
                str,
            ),
        ),
        valid_sampler=dict(
            CLS=FixedBatchSizeBatchSampler,
            batch_size=1,
        ),
        test_datapipe=dict(
            CLS=field(
                HearScenePipe,
                "\nThe first datapipe class to be applied to the corpus. You can add the **kwargs right below this CLS key",
                str,
            ),
        ),
        test_sampler=dict(
            CLS=FixedBatchSizeBatchSampler,
            batch_size=1,
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
            legacy=True,  # FIXME: Leo
        ),
        downstream=dict(
            CLS=field(
                HearFullyConnectedPrediction,
                "\nThe downstream model class for each task. You can add the **kwargs right below this CLS key",
                str,
            ),
            hidden_layers=2,
            pooling="mean",
        ),
        task=dict(
            CLS=field(
                HearScenePredictionTask,
                "\nThe task class defining what to do for each train/valid/test step in the train/valid/test dataloader loop"
                "\nYou can add the **kwargs right below this CLS key",
                str,
            ),
            prediction_type="???",
            scores="???",
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

    @default_cfg(
        **Trainer.train.default_except(
            optimizer=dict(
                CLS="torch.optim.Adam",
                lr=1.0e-3,
            ),
            trainer=dict(
                total_steps=150000,
                log_step=100,
                eval_step=1000,
                save_step=100,
                gradient_clipping=1.0,
                gradient_accumulate_steps=1,
                valid_metric="???",
                valid_higher_better="???",
            ),
        )
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

    @default_cfg(**Trainer.inference.default_cfg)
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @default_cfg(
        **Problem.run.default_except(
            stages=["setup", "train", "inference"],
            start_stage="setup",
            final_stage="inference",
            setup=setup.default_cfg.deselect("workspace", "resume"),
            train=train.default_cfg.deselect("workspace", "resume"),
            inference=inference.default_cfg.deselect("workspace", "resume"),
        )
    )
    @classmethod
    def run(cls, **cfg):
        super().run(**cfg)

    @default_cfg(
        num_fold=field(5, "The number of folds to run cross validation", int),
        **run.default_except(
            workspace=field(
                "???",
                "The root workspace for all folds.\n"
                "Each fold will use a 'fold_{id}' sub-workspace under this root workspace",
            ),
            setup=dict(
                corpus=dict(
                    test_fold=field(
                        "TBD", "This will be auto-set by 'run_cross_validation'"
                    )
                )
            ),
        ),
    )
    @classmethod
    def cross_validation(cls, **cfg):
        """
        Except 'num_fold', all other fields are for 'run' for every fold. That is, all folds shared the same
        config (training hypers, dataset root, etc) except 'workspace' and 'test_fold' are different
        """
        cfg = Container(cfg)
        workspaces = [
            str(Workspace(cfg.workspace) / f"fold_{fold_id}")
            for fold_id in range(cfg.num_fold)
        ]
        for fold_id, workspace in enumerate(workspaces):
            fold_cfg = cfg.clone().deselect("num_fold")

            fold_cfg.workspace = workspace
            fold_cfg.setup.corpus.test_fold = fold_id
            cls.run(
                **fold_cfg,
            )
        metrics = defaultdict(list)
        for fold_id, workspace in enumerate(workspaces):
            workspace = Workspace(workspace)
            metric = workspace["test_metrics"]
            for key, value in metric.items():
                metrics[key].append(value)

        avg_result = dict()
        for key, values in metrics.items():
            avg_score = sum(values) / len(values)
            avg_result[key] = avg_score
            logger.info(f"Average {key}: {avg_score}")

        Workspace(cfg.workspace).put(avg_result, "avg_test_metrics", "yaml")
