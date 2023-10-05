from s3prl.corpus.hear import dcase_2016_task2
from s3prl.util.configuration import default_cfg, field
from s3prl.nn.hear import HearFullyConnectedPrediction
from s3prl.task.event_prediction import EventPredictionTask

from .timestamp import HearTimestamp


class Dcase2016Task2(HearTimestamp):
    @default_cfg(
        **HearTimestamp.setup.default_except(
            corpus=dict(
                CLS=field(
                    dcase_2016_task2,
                    "\nThe corpus class. You can add the **kwargs right below this CLS key",
                    str,
                ),
                dataset_root=field(
                    "???",
                    "The root path of the corpus",
                    str,
                ),
            ),
            downstream=dict(
                CLS=field(
                    HearFullyConnectedPrediction,
                    "\nThe downstream model class for each task. You can add the **kwargs right below this CLS key",
                    str,
                ),
                output_size=11,
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
                scores=["event_onset_200ms_fms", "segment_1s_er"],
                postprocessing_grid={
                    "median_filter_ms": [250],
                    "min_duration": [125, 250],
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
                valid_metric="event_onset_200ms_fms",
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
