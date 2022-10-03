import logging
from collections import defaultdict

from s3prl import Container, Workspace
from s3prl.corpus.hear import maestro
from s3prl.util.configuration import default_cfg, field
from s3prl.nn.hear import HearFullyConnectedPrediction
from s3prl.task.event_prediction import EventPredictionTask

from .timestamp import HearTimestamp

logger = logging.getLogger(__name__)


class Maestro(HearTimestamp):
    @default_cfg(
        **HearTimestamp.setup.default_except(
            corpus=dict(
                CLS=field(
                    maestro,
                    "\nThe corpus class. You can add the **kwargs right below this CLS key",
                    str,
                ),
                dataset_root=field(
                    "???",
                    "The root path of the corpus",
                    str,
                ),
                test_fold=field("???", "The testing fold id. Options: [0, 1, 2, 3, 4]"),
            ),
            downstream=dict(
                CLS=field(
                    HearFullyConnectedPrediction,
                    "\nThe downstream model class for each task. You can add the **kwargs right below this CLS key",
                    str,
                ),
                output_size=87,
                hidden_layers=2,
            ),
            task=dict(
                CLS=field(
                    HearEventPredictionTask,
                    "\nThe task class defining what to do for each train/valid/test step in the train/valid/test dataloader loop"
                    "\nYou can add the **kwargs right below this CLS key",
                    str,
                ),
                prediction_type="multilabel",
                scores=["event_onset_50ms_fms", "event_onset_offset_50ms_20perc_fms"],
                postprocessing_grid={
                    "median_filter_ms": [150],
                    "min_duration": [50],
                },
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        super().setup(**cfg)

    @default_cfg(
        **HearTimestamp.train.default_except(
            optimizer=dict(
                CLS="torch.optim.Adam",
                lr=1.0e-3,
            ),
            trainer=dict(
                total_steps=15000,
                log_step=100,
                eval_step=500,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate_steps=1,
                valid_metric="event_onset_50ms_fms",
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

    @default_cfg(**HearTimestamp.inference.default_cfg)
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @default_cfg(
        **HearTimestamp.run.default_except(
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
