import logging
from collections import defaultdict

from s3prl.base.container import Container
from s3prl.base.workspace import Workspace
from s3prl.corpus.iemocap import iemocap_for_superb
from s3prl.dataset.utterance_classification_pipe import UtteranceClassificationPipe
from s3prl.nn import MeanPoolingLinear
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.utterance_classification_task import UtteranceClassificationTask
from s3prl.util.configuration import default_cfg, field

from .base import SuperbProblem

logger = logging.getLogger(__name__)


class SuperbER(SuperbProblem):
    """
    Superb Emotion Classification problem
    """

    @default_cfg(
        **SuperbProblem.setup.default_except(
            corpus=dict(
                CLS=iemocap_for_superb,
                dataset_root="???",
                test_fold=field(
                    "???",
                    "The session in IEMOCAP used for testing.\n"
                    "The other sessions will be used for training and validation.",
                ),
            ),
            train_datapipe=dict(
                CLS=UtteranceClassificationPipe,
                train_category_encoder=True,
            ),
            train_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=4,
                shuffle=True,
            ),
            valid_datapipe=dict(
                CLS=UtteranceClassificationPipe,
            ),
            valid_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=4,
            ),
            test_datapipe=dict(
                CLS=UtteranceClassificationPipe,
            ),
            test_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=4,
            ),
            downstream=dict(
                CLS=MeanPoolingLinear,
                hidden_size=256,
            ),
            task=dict(
                CLS=UtteranceClassificationTask,
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        """
        This setups the ER problem, containing train/valid/test datasets & samplers and a task object
        """
        super().setup(**cfg)

    @default_cfg(
        **SuperbProblem.train.default_except(
            optimizer=dict(
                CLS="torch.optim.Adam",
                lr=1.0e-4,
            ),
            trainer=dict(
                total_steps=30000,
                log_step=500,
                eval_step=1000,
                save_step=1000,
                gradient_clipping=1.0,
                gradient_accumulate_steps=8,
                valid_metric="accuracy",
                valid_higher_better=True,
            ),
        )
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

    @default_cfg(**SuperbProblem.inference.default_cfg)
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @default_cfg(
        **SuperbProblem.run.default_except(
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
